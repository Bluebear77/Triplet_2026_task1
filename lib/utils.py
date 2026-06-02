import re
import pandas as pd


_WS_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(
    r"\b[^\W\d_][\w]*(?:[A-Z][a-z]+)+\b",
    re.UNICODE
)

# simple number: floatにしやすいもの
SIMPLE_NUMBER_RE = re.compile(
    r"-?(?:\d+|\d{1,3}(?:[,\s]\d{3})+)(?:[.,]\d+)?$"
)

# expanded numeric expressions
YEAR_RANGE_RE = re.compile(r"\d{4}(?:-\d{4})?$")
VOLTAGE_RE = re.compile(r"-?\d+(?:\.\d+)?\s?V$", re.IGNORECASE)
POWER_RE = re.compile(r"-?\d+(?:\.\d+)?\s?mW$", re.IGNORECASE)
PERCENT_RE = re.compile(r"-?\d+(?:[.,]\d+)?\s?%$")
MAGNITUDE_RE = re.compile(r"-?\d+(?:[.,]\d+)?\s?[KMB]$", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:(?:US)?\$)\s?\d+(?:,\d{3})*(?:\.\d+)?$", re.IGNORECASE)

def get_value_types() -> list[str]:
    value_types = ["num"]
    for t in ['year', 'voltage', 'power', 'percent', 'magnitude', 'price']:
        value_types.append(f"num({t})")

    value_types.append("str")
    value_types.append("unknown")
    return value_types

def detect_value_type(text: str) -> str:
    """
    値1つを num / num(year) / str に分類する
    """
    c = str(text).strip()
    if c == "":
        return "unknown"

    # num: simple numeric
    if SIMPLE_NUMBER_RE.fullmatch(c):
        return "num"

    # num: expanded numeric
    if YEAR_RANGE_RE.fullmatch(c):
        return "num(year)"
    if VOLTAGE_RE.fullmatch(c):
        return "num(voltage)"
    if POWER_RE.fullmatch(c):
        return "num(power)"
    if PERCENT_RE.fullmatch(c):
        return "num(percent)"
    if MAGNITUDE_RE.fullmatch(c):
        return "num(magnitude)"
    if PRICE_RE.fullmatch(c):
        return "num(price)"

    return "str"

def is_capitalized(word):
    return word[0].isupper()


def normalize_text(s: str) -> str:
    s = str(s).replace("..\n", ".\n")
    s = s.replace("\n", " ")
    s = re.sub(r"^\s*[*-]+\s*", "", s) # 頭の*を削除
    s = _WS_RE.sub(" ", s).strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = smart_fix(remove_citations(s))
    return s

def word_count(text):
    if not text:
        return 0
    return len(text.split())

def char_count(text):
    return len(text)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

def tokenize(s: str) -> list[str]:
    s = normalize_text(s).lower()
    return _TOKEN_RE.findall(s)


def soft_norm(s: str) -> str:
    """
    Stronger normalization for matching candidate strings to gold.
    """
    s = normalize_text(s).lower()

    # normalize ampersand
    s = s.replace("&", " and ")

    # drop leading 'the '
    s = re.sub(r"^the\s+", "", s)

    # strip punctuation except internal spaces/alnum
    # this can break numerics
    #s = s.translate(str.maketrans("", "", string.punctuation))

    # collapse whitespace again
    s = _WS_RE.sub(" ", s).strip()
    return s

def divide_camel_case_string(txt: str) -> str:
    # e.g. "UpperCaseWordUpperCaseWord"
    return CAMEL_RE.sub(lambda m: " ".join(re.findall(r"[A-Z][a-z]+", m.group(0))), txt)


def norm_annotated_value(s: str) -> str:
    s = divide_camel_case_string(s)
    return soft_norm(s)

def flatten_df(df: pd.DataFrame, table_input_type: str) -> str:
    
    if table_input_type == 'tablellama':
        parts = []

        # 列ヘッダ
        header = " | ".join(df.columns)
        parts.append(f"[TAB] col: | {header} |")

        # 各行
        for i, row in df.iterrows():
            row_str = " | ".join(map(str, row.values))
            parts.append(f"row {i+1}: | {row_str} |")

        # [SEP]で連結
        return " [SEP] ".join(parts)

    elif table_input_type == "structured":
        lines = []
        header = " | ".join(map(str, df.columns))
        lines.append(f"Columns: {header}")
        for i, row in df.iterrows():
            row_str = " | ".join(map(str, row.values))
            lines.append(f"Row {i+1}: {row_str}")
        return "\n".join(lines)

    elif table_input_type == 'text' or table_input_type == 'text_fr':
        sentences = []

        # if df.columns are all numeric
        if all(re.match(r'^\d+$', str(col)) for col in df.columns) and len(df.columns) == 2:
            for _, row in df.iterrows():
                col1, col2 = row.values
                if pd.isna(col1) or pd.isna(col2) or col1 == "" or col2 == "":
                    continue
                if table_input_type == 'text_fr':
                    sentences.append(f"{col1} est {col2}.")
                else:
                    sentences.append(f"{col1} is {col2}.")
            
            return " ".join(sentences)
        else:
            for _, row in df.iterrows():
                parts = []
                for i, col in enumerate(df.columns):
                    val = row.iloc[i]

                    if pd.isna(val) or val == "":
                        continue

                    if table_input_type == 'text_fr':
                        parts.append(f"{col} est {val}")
                    else:
                        parts.append(f"{col} is {val}")

                if parts:
                    sentences.append(", ".join(parts) + ".")

        return " ".join(sentences)

    elif table_input_type == "keyValue":
        df = df.copy()

        def _is_empty(v) -> bool:
            if pd.isna(v):
                return True
            if isinstance(v, str) and v.strip() == "":
                return True
            return False

        def _clean_text(v) -> str:
            if pd.isna(v):
                return ""
            s = str(v).strip()
            s = re.sub(r"\s+", " ", s)
            s = s.rstrip(" ,;:")
            return s

        def _normalize_col(col) -> str:
            s = _clean_text(col)
            # 末尾のコロンや余計な空白を整理
            s = s.rstrip(":")
            return s

        def _looks_like_key_value_table(df: pd.DataFrame) -> bool:
            # 2列で、列名が数値 or 無意味なことが多いケース
            if len(df.columns) != 2:
                return False

            cols_numeric = all(re.fullmatch(r"\d+", str(c).strip()) for c in df.columns)

            # 1列目がキーっぽく、2列目が値っぽいなら key-value とみなす
            first_col = df.iloc[:, 0].dropna().astype(str).str.strip()
            second_col = df.iloc[:, 1].dropna().astype(str).str.strip()

            if len(first_col) == 0 or len(second_col) == 0:
                return cols_numeric

            # 1列目のほうが短い/ラベルっぽいことが多い
            first_avg_len = first_col.map(len).mean()
            second_avg_len = second_col.map(len).mean()

            label_like_ratio = first_col.map(
                lambda x: bool(re.fullmatch(r"[A-Za-z0-9 _/\-()%.]+", str(x)))
            ).mean()

            return cols_numeric or (
                label_like_ratio >= 0.7 and first_avg_len <= second_avg_len
            )

        # 2列 key-value table
        if _looks_like_key_value_table(df):
            sentences = []
            for _, row in df.iterrows():
                key = _clean_text(row.iloc[0])
                val = _clean_text(row.iloc[1])

                if not key or not val:
                    continue

                # "price" / "CPU" のようなキーを自然に保つ
                sentences.append(f"{key}: {val}.")
            return " ".join(sentences)

        # 一般表: 1行を1文として保持
        sentences = []
        for _, row in df.iterrows():
            parts = []
            for col, val in row.items():
                if _is_empty(val):
                    continue

                col_text = _normalize_col(col)
                val_text = _clean_text(val)

                if not col_text or not val_text:
                    continue

                parts.append(f"{col_text}: {val_text}")

            if parts:
                # row境界を保つために ; 区切り
                sentences.append("; ".join(parts) + ".")

        return " ".join(sentences)
    else:
        raise NotImplementedError(f"Unknown table_input_type: {table_input_type}")


def smart_fix(text: str) -> str:
    patterns = [
        (r'([a-z])([A-Z])', r'\1 \2'),
        (r'(\))([A-Za-z])', r'\1 \2'),
        (r"(\d)([A-Za-z])", r"\1 \2"),
        (r',([A-Za-z])', r', \1'),
        (r'([a-z])(\()', r'\1 \2'),
        (r'\.([A-Z])', r'. \1'),
    ]
    
    for p, r in patterns:
        text = re.sub(p, r, text)
    
    return text

def remove_citations(text: str) -> str:
    # [1]の引用を削除
    text = re.sub(r"\s*\[\d+\]", "", text)
    return text.strip()


def extract_relevant_sentences(text: str, subject: str, obj: str, window: int = 1, max_sentences: int = 5) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text)
    hit_ids = []
    for i, s in enumerate(sents):
        sl = s.lower()
        if subject.lower() in sl or obj.lower() in sl:
            hit_ids.append(i)

    selected = []
    used = set()
    for i in hit_ids:
        for j in range(max(0, i - window), min(len(sents), i + window + 1)):
            if j not in used:
                used.add(j)
                selected.append(sents[j])

    if not selected:
        selected = sents[:max_sentences]

    return " ".join(selected[:max_sentences])

def format_document_text(doc: dict) -> str:
    for i, t in enumerate(doc["texts"]):
        if i >= 1:
            break
        value = normalize_text(t["value"])
        title = t.get("title")
        assert i == 0 or value != ""  # 最初のテキストは空であってはならない
        assert i == 0 and title is None # 最初のテキストにtitleがあってはならない
            
    return value


def extract_relevant_subtable(df: pd.DataFrame, subject: str, obj: str, max_rows: int = 3, max_cols: int | None = 4) -> tuple[pd.DataFrame, int]:

    def _extract_relevant_rows_df(
        df: pd.DataFrame,
        subject: str,
        obj: str,
        max_rows: int = 3,
    ) -> tuple[pd.DataFrame, int]:
        subject_l = subject.lower()
        obj_l = obj.lower()

        scores = []
        table_score = 0

        for idx, row in df.iterrows():
            text = " ".join(str(v).lower() for v in row.values)

            score = 0
            if subject_l in text:
                score += 3
            if obj_l in text:
                score += 3

            score += sum(tok in text for tok in subject_l.split())
            score += sum(tok in text for tok in obj_l.split())

            scores.append((score, idx))
            table_score += score

        # スコア順
        scores.sort(reverse=True)

        selected_idx = [idx for score, idx in scores if score > 0][:max_rows]

        # fallback
        if not selected_idx:
            selected_idx = list(df.index[:max_rows])

        return df.loc[selected_idx], table_score

    def _select_relevant_columns(
        df: pd.DataFrame,
        subject: str,
        obj: str,
        max_cols: int = 4,
    ) -> tuple[pd.DataFrame, int]:
        subject_l = subject.lower()
        obj_l = obj.lower()

        col_scores = []
        table_score = 0

        for col in df.columns:
            col_l = col.lower()
            score = 0

            # 列名マッチ
            if subject_l in col_l:
                score += 2
            if obj_l in col_l:
                score += 2

            # 列内セルマッチ
            col_text = " ".join(str(v).lower() for v in df[col].values)
            if subject_l in col_text:
                score += 3
            if obj_l in col_text:
                score += 3

            col_scores.append((score, col))
            table_score += score

        col_scores.sort(reverse=True)

        selected_cols = [c for s, c in col_scores if s > 0][:max_cols]

        if not selected_cols:
            selected_cols = list(df.columns[:max_cols])

        return df[selected_cols], table_score


    df_sub = df.copy()
    score = 0
    if max_rows is not None:
        df_sub, score1 = _extract_relevant_rows_df(df_sub, subject, obj, max_rows)
        score += score1
    if max_cols is not None:
        df_sub, score2 = _select_relevant_columns(df_sub, subject, obj, max_cols)
        score += score2
    return df_sub, score
