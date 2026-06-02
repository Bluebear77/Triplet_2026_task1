import re
from nltk.corpus import stopwords

from .utils import (
    normalize_text,
    detect_value_type,
    CAMEL_RE,    
)

stop_words = set(stopwords.words("english"))

'''
# 大文字で始まる単語を最大6語まで連続で取る(最低2語以上)
CAP_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){1,5}\b"
)
# Camel case (例UppercaseWordUppercaseWord)
CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")

SINGLE_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9.'-]{1,}\b")
'''

## unicode対応版
CAP_PHRASE_RE = re.compile(
    r"\b[^\W\d_][\w&.'-]*(?:\s+[^\W\d_][\w&.'-]*){1,5}\b",
    re.UNICODE
)

SINGLE_TOKEN_RE = re.compile(
    r"\b[^\W\d_][\w.'-]{1,}\b",
    re.UNICODE
)

CORP_SUFFIXES = [
    " inc", " inc.", " llc", " ltd", " ltd.", " corporation", " corp", " corp.",
    " company", " co", " co.", " group", " studios", " entertainment",
    "production", " productions", " pictures", " communications", " networks", "television",
    "cinema", "film", "films", "bank", "mobile"
]

def extract_capitalized_phrases_ascii(doc):
    vals = []
    for t in doc.get("texts", []) or []:
        txt = normalize_text(t.get("value", ""))
        vals.extend(CAP_PHRASE_RE.findall(txt))
    return [normalize_text(v) for v in vals if normalize_text(v)]


LOWER_LINKERS = {
    "de", "du", "des", "d", "d'",
    "la", "le", "les", "et",
    "van", "von", "di", "da", "del"
}

ORDINAL_TOKENS = {"ier", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}

TOKEN_RE = re.compile(r"[^\W_]+(?:['.-][^\W_]+)*", re.UNICODE)

def is_capitalized_token(tok: str) -> bool:
    return bool(tok) and tok[0].isalpha() and tok[0].isupper()

def is_roman_or_ordinal(tok: str) -> bool:
    low = tok.lower()
    return low in ORDINAL_TOKENS or bool(re.fullmatch(r"[IVXLCDM]+", tok))

def extract_phrases_from_text(txt: str, max_len: int = 12):
    # e.g. "The University of California, Los Angeles (UCLA)"
    toks = TOKEN_RE.findall(txt)
    out = []
    n = len(toks)

    for i in range(n):
        first = toks[i]
        if not is_capitalized_token(first):
            continue

        phrase = [first]

        for j in range(i + 1, min(i + max_len, n)):
            tok = toks[j]
            low = tok.lower()

            if (
                is_capitalized_token(tok)
                or is_roman_or_ordinal(tok)
                or low in LOWER_LINKERS
                or re.fullmatch(r"\d{4}", tok)  # Euro 1995 みたいなの用
            ):
                phrase.append(tok)
            else:
                break

        out.append(" ".join(phrase))

    return out

def extract_capitalized_phrases_unicode(doc):
    # e.g. "The University of California, Los Angeles (UCLA)"
    vals = []
    for t in doc.get("texts", []) or []:
        txt = normalize_text(t.get("value", ""))
        vals.extend(extract_phrases_from_text(txt))
    return [normalize_text(v) for v in vals if normalize_text(v)]

def extract_camel_case(doc):
    # e.g. "UpperCaseWordUpperCaseWord"
    vals = []
    for t in doc.get("texts", []) or []:
        txt = normalize_text(t.get("value", ""))
        vals.extend(CAMEL_RE.findall(txt))
    return [normalize_text(v) for v in vals if normalize_text(v)]

def extract_single_token_proper_nouns(doc):
    # e.g. "UCLA"
    vals = []
    for t in doc.get("texts", []) or []:
        txt = normalize_text(t.get("value", ""))
        vals.extend(SINGLE_TOKEN_RE.findall(txt))
    return [normalize_text(v) for v in vals if normalize_text(v)]


LOWER_NP_STARTERS = {
    "le", "la", "les", "un", "une", "des",
    "du", "de", "d'", "première", "dernier", "dernière"
}

def extract_lowercase_noun_phrases(doc):
    vals = []
    for t in doc.get("texts", []) or []:
        txt = normalize_text(t.get("value", ""))
        toks = TOKEN_RE.findall(txt)
        n = len(toks)

        for i in range(n):
            low = toks[i].lower()
            if low not in LOWER_NP_STARTERS and not low[0].islower():
                continue

            phrase = [toks[i]]
            for j in range(i + 1, min(i + 8, n)):
                tok = toks[j]
                low2 = tok.lower()

                if (
                    tok[0].islower()
                    or is_capitalized_token(tok)
                    or low2 in LOWER_LINKERS
                ):
                    phrase.append(tok)
                else:
                    break

            vals.append(" ".join(phrase))

    return [normalize_text(v) for v in vals if normalize_text(v)]


def expand_table_cell(cell: str, substring_append: bool):
    cell = normalize_text(cell)
    out = [cell]

    # trailing number removal
    stripped = re.sub(r"\s+\d+$", "", cell).strip()
    if stripped and stripped != cell:
        out.append(stripped)

    # corporate suffix removal
    low = cell.lower()
    for suf in CORP_SUFFIXES:
        if low.endswith(suf):
            base = cell[:len(cell) - len(suf)].strip()
            if base:
                out.append(base)

    # possessive removal
    if cell.endswith("'s"):
        out.append(cell[:-2].strip())

    if substring_append:
        toks = cell.split()

        # person last name only
        if 2 <= len(toks) <= 4 and all(re.match(r"[A-Z][a-zA-Z.'-]+$", x) for x in toks):
            out.append(toks[-1])

        # NEW: drop first token once
        if len(toks) >= 2:
            out.append(" ".join(toks[1:]))

        # NEW: drop last token once
        if len(toks) >= 2:
            out.append(" ".join(toks[:-1]))

    return [x for x in out if x]


def expand_person_tokens(candidates):
    extra = []

    for c in candidates:
        toks = c.split()
        #for t in toks:
        #    if re.match(r"[A-Z][a-z]+$", t):
        #        # e.g., "Gary Oldman" -> "Gary", "Oldman"
        #        extra.append(t)
        # 副作用: bank of Kenya -> Bank of, Kenya

        if c.endswith("'s"):
            # e.g., "Gary Oldman's" -> "Gary Oldman"
            base = c[:-2]
            if base:
                extra.append(base)

    return extra


def expand_company_variants(candidates):
    extra = []

    for c in candidates:
        c_norm = normalize_text(c)
        low = c_norm.lower()

        # strip trailing numbers
        # e.g. "AMC Networks 14" -> "AMC Networks"
        stripped = re.sub(r"\s+\d+$", "", c_norm).strip()
        if stripped and stripped != c_norm:
            extra.append(stripped)

        for suf in CORP_SUFFIXES:
            if low.endswith(suf):
                # e.g., "Warner Bros. Pictures" -> "Warner Bros."
                base = c_norm[:len(c_norm) - len(suf)].strip()
                if base:
                    extra.append(base)

    return extra

def contains_non_stopwords(text:str) -> bool:
    words = text.lower().split()
    filtered = [w for w in words if w not in stop_words]
    return len(filtered) > 0

def is_valid_positive_candidate(c: str, allow_number: bool) -> bool:

    if len(c) > 100:
        ## too long candidates are unlikely to be correct
        return False

    if allow_number and 'num' in detect_value_type(c):
        return True

    if not contains_non_stopwords(c):
        return False

    # too long single-token candidates are unlikely to be correct
    if len(c.split()) > 20:
        return False

    c = normalize_text(c)
    if not c:
        return False

    # only remove obvious junk
    if re.fullmatch(r"[\d\W]+", c):
        return False
    if re.fullmatch(r"[A-Z]\.", c):
        return False

    return True


def is_valid_negative_candidate(c: str, allow_number: bool) -> bool:
    """
    Precision-oriented filter for negative sampling.
    Avoid junk like 'The', 'Gary', 'Players', etc.
    """
    if len(c) > 100:
        ## too long candidates are unlikely to be correct
        return False
    
    if allow_number and 'num' in detect_value_type(c):
        return True
    
    if not contains_non_stopwords(c):
        return False
    
    c = normalize_text(c)
    if not c:
        return False
    
    if len(c) <= 2:
        return False
    if re.fullmatch(r"[\d\W]+", c):
        return False
    if re.fullmatch(r"[A-Z]\.", c):
        return False

    toks = c.split()

    # 2+ token candidates are usually fine
    if 2 <= len(toks):
        return True

    # too long single-token candidates are unlikely to be correct
    if len(toks) > 20:
        return False

    # 1-token candidates: be much stricter
    t = toks[0]
    low = t.lower()

    if len(t) < 4:
        return False

    if CAMEL_RE.fullmatch(t):
        return True

    # allow things like AMC-14
    if re.fullmatch(r"[A-Z0-9][A-Za-z0-9-]{3,}", t):
        return True

    # keep longer proper nouns like Davis
    return t[0].isupper() and len(t) >= 5
