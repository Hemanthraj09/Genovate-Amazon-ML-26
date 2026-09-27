"""GPU cross-encoder for the uncertain pairs (optional, needs torch + transformers).

A small pretrained multilingual encoder (intfloat/multilingual-e5-small, MIT,
118M parameters) reads the RAW name and address of both records together and is
fine-tuned to say whether they are the same business. It only sees the pairs the
tree models are unsure about -- stage-1 or stage-2 probability in (0.01, 0.99) --
which is where near-twin decoys, French wording and transliteration live.

Honest scores: the band pairs are split by the tree models' fold id into halves
A (folds 0-1) and B (folds 2-3). Model A trains on A and scores B, model B the
reverse, and test gets the mean of both. The blend with the tree probability is
then tuned on those out-of-half scores (see ce_blend.py).

    BER_VARIANT=fix BER_MODEL_TAG=v4bh <venv-python> ce.py <model dir>
Outputs WORK_DIR/ce/{oof,test}_ce.parquet with columns qid, s1, ce (a logit).
"""
import math
import os
import sys
import time
import numpy as np
import polars as pl
import torch
import config as C
import blocking as B
import train as TR

LO, HI = 0.01, 0.99
MAX_LEN = 96
BATCH = 64
LR = 5e-5
MAX_TRAIN = int(os.environ.get("BER_CE_MAX_TRAIN", "350000"))   # pairs per half


def band(tag, split):
    """Band pairs of a split with stage-1 and stage-2 probability."""
    n1, n2 = ("oof", "oof2") if split == "train" else ("test_pred", "test_pred2")
    p1 = pl.read_parquet(C.work(tag, f"{n1}.parquet")).select("qid", "s1", pl.col("p").alias("p1"))
    p2 = pl.read_parquet(C.work(tag, f"{n2}.parquet"))
    keep = ["qid", "s1", "p"] + (["y"] if "y" in p2.columns else [])
    j = p2.select(keep).join(p1, on=["qid", "s1"])
    inb = lambda c: (pl.col(c) > LO) & (pl.col(c) < HI)
    return j.filter(inb("p") | inb("p1"))


def texts(pairs, split):
    """Raw 'name | address' strings for both sides of each pair."""
    def fmt(df):
        return (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
    s1 = pl.read_parquet(C.work("raw", f"{split}_s1.parquet"), columns=["idx", "business_name", "business_address"])
    s1 = s1.select(pl.col("idx").alias("s1"), fmt(s1).alias("ts"))
    qs = []
    for src in (2, 3):
        q = pl.read_parquet(C.work("raw", f"{split}_s{src}.parquet"), columns=["idx", "business_name", "business_address"])
        qs.append(q.select(B.qid_expr(src).alias("qid"), fmt(q).alias("tq")))
    return pairs.join(pl.concat(qs), on="qid", how="left").join(s1, on="s1", how="left")


def batches(df, bs, shuffle=False, seed=0):
    idx = np.arange(df.height)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield df[idx[i:i + bs]]


def encode(tok, b, dev):
    e = tok(b["tq"].to_list(), b["ts"].to_list(), truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in e.items()}


def fit(mdir, df, dev, seed):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1).to(dev)
    # the 250k x 384 word-embedding table is most of the parameters; its AdamW
    # state alone ran a 6 GB laptop GPU out of memory. Keep it frozen.
    m.bert.embeddings.word_embeddings.weight.requires_grad_(False)
    if df.height > MAX_TRAIN:
        df = df.sample(MAX_TRAIN, seed=seed)
    steps = math.ceil(df.height / BATCH)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    m.train(); t0 = time.time(); run = 0.0
    for i, b in enumerate(batches(df, BATCH, shuffle=True, seed=seed)):
        try:
            e = encode(tok, b, dev)
            y = torch.tensor(b["y"].to_numpy(), dtype=torch.float32, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(**e).logits.squeeze(-1)
            loss = lossf(out.float(), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        except (torch.OutOfMemoryError, RuntimeError) as err:
            # the GPU is shared with the desktop; if another app grabs memory,
            # drop this batch rather than the whole run
            print(f"  step {i}: skipped batch ({str(err)[:60]})", flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); sch.step(); continue
        sch.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item()
        if i % 500 == 0:
            print(f"  step {i}/{steps} loss {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return tok, m


@torch.no_grad()
def score(tok, m, df, dev, bs=256):
    m.eval(); out = []
    for b in batches(df, bs):
        for tries in range(6):
            try:
                parts = [b[i:i + max(1, bs >> tries)] for i in range(0, b.height, max(1, bs >> tries))]
                got = []
                for pb in parts:
                    e = encode(tok, pb, dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        got.append(m(**e).logits.squeeze(-1).float().cpu().numpy())
                out.extend(got)
                break
            except (torch.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()     # retry this batch in smaller pieces
        else:
            raise RuntimeError("scoring kept running out of GPU memory")
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def prep():
    """Write the band pairs and their texts once (main env), so the GPU process
    stays small: on Windows the driver needs system RAM too, and a 5 GB process
    next to the LightGBM jobs ran it out."""
    out = C.work("ce", "x").parent
    tr = band(C.MODEL_DIR, "train")
    tr = TR.add_folds(tr.lazy()).select("qid", "s1", "y", "fold").collect()
    texts(tr, "train").write_parquet(out / f"band_train_{C.MODEL_TAG}.parquet")
    texts(band(C.MODEL_DIR, "test").select("qid", "s1"), "test").write_parquet(out / f"band_test_{C.MODEL_TAG}.parquet")


def main(mdir):
    dev = "cuda"
    t0 = time.time()
    out = C.work("ce", "x").parent
    tr = pl.read_parquet(out / f"band_train_{C.MODEL_TAG}.parquet")
    te = pl.read_parquet(out / f"band_test_{C.MODEL_TAG}.parquet")
    print(f"band pairs: train {tr.height:,}  test {te.height:,} ({time.time() - t0:.0f}s)", flush=True)
    half = (pl.col("fold") >= 2)
    for h, seed in ((0, 11), (1, 12)):
        done = out / f"half{h}_{C.MODEL_TAG}.parquet"
        if done.exists():                    # resume: each half is saved as it finishes
            print(f"half {h}: already done", flush=True)
            continue
        fit_df, pred_df = tr.filter(half == bool(h)), tr.filter(half != bool(h))
        print(f"half {h}: fit {fit_df.height:,} -> score {pred_df.height:,}", flush=True)
        tok, m = fit(mdir, fit_df, dev, seed)
        o = pred_df.select("qid", "s1").with_columns(pl.Series("ce", score(tok, m, pred_df, dev)))
        t = te.select("qid", "s1").with_columns(pl.Series("ce", score(tok, m, te, dev)))
        pl.concat([o.with_columns(pl.lit("oof").alias("kind")), t.with_columns(pl.lit("test").alias("kind"))]).write_parquet(done)
        print(f"half {h} done ({time.time() - t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()
    hs = [pl.read_parquet(out / f"half{h}_{C.MODEL_TAG}.parquet") for h in (0, 1)]
    pl.concat([x.filter(pl.col("kind") == "oof") for x in hs]).drop("kind").write_parquet(out / f"oof_ce_{C.MODEL_TAG}.parquet")
    t = (hs[0].filter(pl.col("kind") == "test").drop("kind")
           .join(hs[1].filter(pl.col("kind") == "test").drop("kind"), on=["qid", "s1"], suffix="_b")
           .select("qid", "s1", ((pl.col("ce") + pl.col("ce_b")) / 2).alias("ce")))
    t.write_parquet(out / f"test_ce_{C.MODEL_TAG}.parquet")
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else main(sys.argv[1])
