"""Text normalization for business names and addresses.

Everything here is country-agnostic: the same rules run on every record,
whatever its `country` label (open set -- France is never special-cased).

Knowledge sources (no external data):
  * small hand-written abbreviation / legal-form lists (US, India, French),
  * substitution maps and an Indic-script word dictionary *learned from the
    provided training pairs* (see learn_maps.py), loaded via `load_maps()`,
  * a stdlib (unicodedata) character-level transliterator as fallback for
    Indic-script words the learned dictionary has never seen.
"""
import json
import re
import unicodedata

# ---------------------------------------------------------------- constants
# Leetspeak-style digit substitutions observed in the training pairs.
LEET = str.maketrans("01568", "olsgb")

# Legal forms -> canonical token. Dots/spaces are removed before lookup.
LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp", "pc": "pc",
    "pllc": "pllc", "plc": "plc", "pa": "pa", "privatelimited": "pvtltd",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "ei": "ei", "scop": "scop", "gie": "gie",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
HONORIFIC = {"mr", "mrs", "ms", "dr", "smt", "sri", "shri", "shree", "m/s", "ms/",
             "the", "messrs"}
STOP_NAME = {"and", "&", "of", "+", "|", "-", "le", "la", "les", "du", "de", "des",
             "et", "l", "d"}
# Generic words the noise generator appends; dropped only in the "strict" core.
GENERIC = {"center", "centre", "services", "service", "partners", "group", "co",
           "company", "holdings", "india", "france", "usa", "us"}
DBA_RE = re.compile(
    r"\b(?:doing business as|d\.b\.a\.?|d/b/a|dba:?|f/k/a|fka|formerly known as|"
    r"formerly|a/k/a|aka|trading as|t/a)(?=\s|$|:)", re.I)
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co|biz|info|fr|us|io)\b")
IDTAG_RE = re.compile(r"\(\s*id\s*:\s*\d+\s*\)|#\d+\b", re.I)
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme|ème)$")
INDIC_RE = re.compile(r"[ऀ-ൿ]")

# Address abbreviation groups (hand-written, generic). First item = canonical.
_ADDR_GROUPS = [
    ["st", "street", "saint", "str"], ["rd", "road"], ["ave", "av", "avenue", "avanue"],
    ["blvd", "bd", "boulevard", "boul", "boulevrd"], ["dr", "drive"], ["ln", "lane"],
    ["ct", "court"], ["cir", "circle"], ["pl", "place"], ["pkwy", "parkway"],
    ["hwy", "highway"], ["trl", "trail"], ["twp", "township"], ["ter", "terrace"],
    ["sq", "square"], ["pt", "point"], ["mt", "mount"], ["ft", "fort"],
    ["n", "north"], ["s", "south"], ["e", "east"], ["w", "west"],
    ["unit", "apt", "apartment", "suite", "ste", "floor", "flr"],
    ["rue", "r"], ["imp", "impasse"], ["ch", "chemin"], ["all", "allee"],
    ["rte", "route"], ["crs", "cours"], ["fbg", "faubourg"], ["qu", "quai"],
    ["nagar", "ngr"], ["bldg", "building"], ["opp", "opposite"], ["nr", "near"],
    ["sec", "sector"], ["grd", "ground"], ["colony", "col"],
]
ADDR_ABBR = {w: g[0] for g in _ADDR_GROUPS for w in g}
# Tokens that carry no identity (house-number prefixes, placeholders).
ADDR_DROP = {"no", "nos", "h", "hno", "h.no", "hn", "door", "number", "num", "n°",
             "null", "<null>", "n/a", "na", "none", "nil", "1/2", "cdp", "of", "the", "city",
             "de", "du", "des", "la", "le", "les", "d", "l"}
POBOX_RE = re.compile(r"^(?:p\.?\s*o\.?\s*box|pmb|box)\s*#?\s*\w+$")

# Learned maps (filled by load_maps()). addr_* maps are keyed by country label;
# a label never seen in training (e.g. France) simply gets no learned map.
MAPS = {"addr_tok": {}, "addr_comp": {}, "indic_word": {}}


def load_maps(path):
    """Load the maps learned from training pairs (see learn_maps.py)."""
    with open(path, encoding="utf-8") as f:
        MAPS.update(json.load(f))


# ------------------------------------------------------ transliteration
def _build_indic_table():
    """Map each Indic code point to (kind, roman) using unicodedata names.

    kind: C=consonant, V=independent vowel, M=vowel sign (matra),
          X=virama, N=nasal (anusvara/candrabindu), H=visarga, D=digit, Z=ignore.
    """
    vowels = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u",
              "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "E": "e",
              "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au", "SHORT E": "e",
              "SHORT O": "o", "CANDRA E": "e", "CANDRA O": "o", "SHORT A": "a"}
    simp = {"tt": "t", "tth": "th", "dd": "d", "ddh": "dh", "nn": "n", "ny": "n",
            "ng": "n", "ss": "sh", "ll": "l", "lll": "l", "rr": "r", "c": "ch",
            "nnn": "n", "zh": "l"}
    table = {}
    for cp in range(0x0900, 0x0D80):
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if not name:
            continue
        if " DIGIT " in name:
            table[ch] = ("D", str(unicodedata.digit(ch, 0)))
        elif " VOWEL SIGN " in name:
            table[ch] = ("M", vowels.get(name.split(" VOWEL SIGN ", 1)[1], ""))
        elif " LETTER " in name:
            base = name.split(" LETTER ", 1)[1]
            if base in vowels:
                table[ch] = ("V", vowels[base])
            else:
                r = base.lower().split()[0]
                r = r[:-1] if r.endswith("a") and len(r) > 1 else r
                table[ch] = ("C", simp.get(r, r))
        elif name.endswith("VIRAMA"):
            table[ch] = ("X", "")
        elif "ANUSVARA" in name or "CANDRABINDU" in name:
            table[ch] = ("N", "n")
        elif "VISARGA" in name:
            table[ch] = ("H", "h")
        else:
            table[ch] = ("Z", "")
    return table


_INDIC = _build_indic_table()


def translit_word(w):
    """Character-level romanization of one Indic-script word (fallback only)."""
    out, pending = [], False
    for ch in w:
        kind, val = _INDIC.get(ch, (None, ch))
        if kind == "C":
            if pending:
                out.append("a")
            out.append(val)
            pending = True
        elif kind == "M":
            out.append(val)
            pending = False
        elif kind == "V":
            if pending:
                out.append("a")
            out.append(val)
            pending = False
        elif kind == "X":
            pending = False
        elif kind in ("N", "H"):
            if pending:
                out.append("a")
            out.append(val)
            pending = False
        elif kind in ("Z",) or ch in "‌‍":
            continue
        else:
            if pending:
                out.append("a")
            pending = False
            out.append(val if kind == "D" else ch)
    return "".join(out)


def translit_text(s):
    """Replace every Indic-script word in `s` by its Latin form.

    Uses the learned word dictionary first (exact inverse of the noise
    generator on words seen in training), else the character fallback.
    """
    if not INDIC_RE.search(s):
        return s
    d = MAPS["indic_word"]
    parts = []
    for w in s.split():
        if INDIC_RE.search(w):
            core = w.strip(".,()[]")
            lat = d.get(core) or translit_word(core)
            parts.append(w.replace(core, lat) if core else w)
        else:
            parts.append(w)
    return " ".join(parts)


# --------------------------------------------------------------- helpers
def strip_accents(s):
    """NFKD-decompose and drop combining marks (é->e, Nº->No)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def fix_leet(tok):
    """Undo digit-for-letter substitutions inside alphabetic words (cardi0logy)."""
    if tok.isdigit() or ORDINAL_RE.match(tok) or not any(c.isalpha() for c in tok):
        return tok
    if not any(c.isdigit() for c in tok):
        return tok
    return tok.translate(LEET)


def dedupe(tokens):
    """Drop repeated tokens, keeping first occurrences (Southern Southern ...)."""
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def consonant_key(tok):
    """Vowel-free skeleton with repeats collapsed; bridges transliteration gaps."""
    s = re.sub(r"[aeiouyhw]", "", tok)
    return re.sub(r"(.)\1+", r"\1", s)


# ------------------------------------------------------------------ names
def _name_tokens(s):
    """Lowercase tokens with punctuation cleaned; legal forms canonicalized."""
    s = s.lower().replace("m/s", " ")
    s = re.sub(r"\b((?:[a-z]\.){2,}[a-z]?)", lambda m: m.group(1).replace(".", ""), s)  # l.l.c. -> llc
    s = s.replace("&", " and ").replace("+", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[\(\)\[\]\{\}<>\"*|_~^=:;!?]", " ", s)
    s = re.sub(r"(?<!\w)['`]|['`](?!\w)", " ", s)
    s = re.sub(r"[,.]", " ", s)
    s = s.replace("'", "")
    toks = []
    for t in s.split():
        t = t.strip("-/#@")
        if not t:
            continue
        toks.append(fix_leet(t))
    return toks


def _split_core(tokens):
    """Split tokens into (core tokens, sorted legal-form set)."""
    core, legal = [], set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # 'private limited' / 'pvt ltd' as a pair
        if t in LEGAL:
            legal.add(LEGAL[t])
        elif t in HONORIFIC or t in STOP_NAME:
            pass
        else:
            core.append(t)
        i += 1
    return dedupe([c for c in core if c]), sorted(legal)


def norm_name(raw):
    """Normalize a business name.

    Returns dict with:
      nm      core tokens (legal forms/honorifics/stopwords removed), space-joined
      nm_s    'strict' core (also without generic appended words)
      nm_cmp  core joined without spaces (for domain/handle comparison)
      nm_alt  core of the other side of a DBA/FKA split ('' if none)
      nm_leg  canonical legal forms, sorted, space-joined
      flags   bitmask: 1=domain, 2=handle, 4=indic script, 8=dba split, 16=id tag
    """
    flags = 0
    s = raw.strip()
    if INDIC_RE.search(s):
        flags |= 4
        s = translit_text(s)
    s = strip_accents(s)
    if IDTAG_RE.search(s):
        flags |= 16
        s = IDTAG_RE.sub(" ", s)
    s = re.sub(r"^[^\w@#]+", "", s).strip()           # leading junk (***, >>, --, ...)
    low = s.lower()
    alt = ""
    m = DBA_RE.search(low)
    if m and low[:m.start()].strip() and low[m.end():].strip():
        before, after = low[:m.start()], low[m.end():]
        b_core, _ = _split_core(_name_tokens(before))
        if b_core:                                      # real DBA: keep name after marker
            flags |= 8
            alt = " ".join(_split_core(_name_tokens(before + " " + after))[0])
            low = after
    dm = None
    for w in low.split():                                # domain anywhere (after honorifics etc.)
        dm = DOMAIN_RE.match(w.strip("|,;()[]"))
        if dm:
            break
    if dm:
        flags |= 1
        stem = dm.group(1).replace("-", "")
        stem = fix_leet(stem)
        return {"nm": stem, "nm_s": stem, "nm_cmp": stem, "nm_alt": alt, "nm_leg": "",
                "flags": flags}
    bare = [w for w in low.split() if w not in HONORIFIC]
    if len(bare) == 1 and bare[0][:1] in "@#":
        low = bare[0]
        flags |= 2
        stem = fix_leet(re.sub(r"[^a-z0-9]", "", low))
        return {"nm": stem, "nm_s": stem, "nm_cmp": stem, "nm_alt": alt, "nm_leg": "",
                "flags": flags}
    core, legal = _split_core(_name_tokens(low))
    strict = [t for t in core if t not in GENERIC] or core
    return {"nm": " ".join(core), "nm_s": " ".join(strict), "nm_cmp": "".join(core),
            "nm_alt": alt, "nm_leg": " ".join(legal), "flags": flags}


# -------------------------------------------------------------- addresses
def _addr_basic_components(raw):
    """Lowercased, accent-free, transliterated address components (comma split)."""
    s = translit_text(raw) if INDIC_RE.search(raw) else raw
    s = strip_accents(s).lower()
    comps = [c.strip() for c in s.split(",")]
    return [c for c in comps if c]


def _addr_comp_tokens(comp):
    """Tokenize one address component with the hand-written abbreviation map."""
    comp = re.sub(r"[#()\[\]\"*]", " ", comp)
    comp = comp.replace(".", " ").replace("&", " and ")
    out = []
    for t in comp.split():
        t = t.strip("-/'")
        if not t:
            continue
        t = ADDR_ABBR.get(t, t)
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
    return out


def norm_address(raw, country=""):
    """Normalize an address.

    Each comma component is tokenized (hand-written abbreviations), then the
    learned component map for this country is tried on the whole component,
    else the learned token map on each token.

    Returns dict with:
      ad      normalized tokens, component order kept, space-joined
      ad_nums digit runs (leading zeros stripped) in order, space-joined
      ad_hn   primary house number (first digit run outside unit/PO-box parts)
      ad_unit tokens of unit / PO-box components
    """
    if not raw or not raw.strip():
        return {"ad": "", "ad_nums": "", "ad_hn": "", "ad_unit": ""}
    cm = MAPS["addr_comp"].get(country, {})
    tm = MAPS["addr_tok"].get(country, {})
    toks, unit = [], []
    for comp in _addr_basic_components(raw):
        c2 = comp.replace(".", "").strip()
        ct = _addr_comp_tokens(comp)
        if not ct:
            continue
        key = " ".join(ct)
        ct = cm[key].split() if key in cm else [tm.get(t, t) for t in ct]
        if not ct:
            continue
        if POBOX_RE.match(c2) or ct[0] in ("unit", "pmb", "box"):
            unit.extend(ct)
            continue
        toks.extend(ct)
    nums = []
    for t in toks:
        for d in re.findall(r"\d+", t):
            d = d.lstrip("0") or "0"
            nums.append(d)
    return {"ad": " ".join(toks), "ad_nums": " ".join(nums),
            "ad_hn": nums[0] if nums else "", "ad_unit": " ".join(unit)}
