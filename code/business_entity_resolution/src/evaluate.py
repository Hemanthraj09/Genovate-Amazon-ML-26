"""Macro F0.5 exactly as defined by the challenge.

Per Source-1 entity:
    G = true matches, k = predicted matches, TP = correct predictions
    G == 0 and k == 0  -> 1.0         (correct singleton)
    G == 0 and k  > 0  -> 0.0         (false merge on a singleton)
    G  > 0 and TP == 0 -> 0.0
    otherwise          -> 1.25*P*R / (0.25*P + R) = 1.25*TP / (k + 0.25*G)
The score is the mean over ALL Source-1 entities in the evaluation set.
"""
import polars as pl


def macro_f05(pred, truth, s1_ids):
    """Macro F0.5.

    pred:   DataFrame (s1, qid) of predicted matches (any other columns ignored)
    truth:  DataFrame (s1, qid) of true matches
    s1_ids: Series/list of every S1 entity in the evaluation set
    Returns (score, per-entity DataFrame with G, k, TP, f).
    """
    ent = pl.DataFrame({"s1": s1_ids}).with_columns(pl.col("s1").cast(pl.UInt32))
    pred = pred.select(pl.col("s1").cast(pl.UInt32), pl.col("qid").cast(pl.UInt32)).unique()
    truth = truth.select(pl.col("s1").cast(pl.UInt32), pl.col("qid").cast(pl.UInt32))
    k = pred.group_by("s1").agg(pl.len().alias("k"))
    g = truth.group_by("s1").agg(pl.len().alias("G"))
    tp = pred.join(truth, on=["s1", "qid"]).group_by("s1").agg(pl.len().alias("TP"))
    e = (ent.join(g, on="s1", how="left").join(k, on="s1", how="left")
            .join(tp, on="s1", how="left").fill_null(0))
    e = e.with_columns(
        pl.when((pl.col("G") == 0) & (pl.col("k") == 0)).then(1.0)
          .when(pl.col("TP") == 0).then(0.0)
          .otherwise(1.25 * pl.col("TP") / (pl.col("k") + 0.25 * pl.col("G")))
          .alias("f"))
    return e["f"].mean(), e


def _self_test():
    """Check the worked example from the problem statement (expected 0.714)."""
    pred = pl.DataFrame({"s1": [1, 1, 1], "qid": [47, 193, 812]})
    truth = pl.DataFrame({"s1": [1, 1], "qid": [47, 812]})
    s, _ = macro_f05(pred, truth, [1])
    assert abs(s - 0.7142857) < 1e-4, s
    # singleton predicted empty = 1.0; singleton with a prediction = 0.0
    s, _ = macro_f05(pl.DataFrame({"s1": [3], "qid": [9]}), truth, [1, 2, 3])
    assert abs(s - (0.0 + 1.0 + 0.0) / 3) < 1e-9, s
    print("evaluate self-test OK")


if __name__ == "__main__":
    _self_test()
