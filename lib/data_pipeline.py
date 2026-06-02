import os
import csv
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from collections import Counter, defaultdict

from .utils import (
    normalize_text,
    soft_norm,
)

from .candidate_func import (
    extract_camel_case,
    extract_capitalized_phrases_ascii,
    extract_capitalized_phrases_unicode,
    extract_single_token_proper_nouns,
    extract_lowercase_noun_phrases,
    expand_table_cell,
    expand_company_variants,
    expand_person_tokens,
    is_valid_positive_candidate,
    is_valid_negative_candidate,
    detect_value_type,
)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_annotations(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "document_id": r["document_id"],
                "subject": normalize_text(r["subject"]),
                "predicate": r["predicate"],
                "object": normalize_text(r["object"].rstrip('.')),
            })
    return rows


def load_properties(path: str, use_short_verb: bool = False, use_first_description: bool = True)  -> List[Dict[str, str]]:
    rows = []
    short_verbs = {}
    if use_short_verb:
        with open(os.path.join(os.path.dirname(path), "short_verb.json"), "r", encoding="utf-8") as f:
            short_verbs = json.load(f)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if use_short_verb:
                description = "e1 " + short_verbs[r["pid"]] + " e2."
            else:
                description = normalize_text(r["description"].split('.')[0]) if use_first_description else normalize_text(r["description"])
            rows.append({
                "pid": r["pid"],
                "label": r["label"],
                "description": description,
                "obj_type": r["obj_type"].split('/'),
            })
    return rows

def is_header_broken(header: list[str]) -> bool:
    if len(header) == 1:
        return False
    # all header cells are empty or None
    if all(cell is None or cell.strip() == "" for cell in header):
        return True
    # all header cells are the same
    if len(set(cell.strip() for cell in header if cell is not None and cell.strip() != "")) == 1:
        return True
    # all header cells are numeric and start from 0,1,2,... and length >= 3
    if all(re.match(r"^\d+$", cell.strip()) for cell in header if cell is not None and cell.strip() != ""):
        nums = sorted(int(cell.strip()) for cell in header if cell is not None and cell.strip() != "")
        if nums == list(range(len(nums))) and len(nums) >= 3:
            return True
    return False


def merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    def is_empty_col(s):
        return ((s.isna()) | (s == "")).all()
    
    df = df.copy()
    i = 0

    while i < df.shape[1] - 1:
        left = df.iloc[:, i]
        right = df.iloc[:, i + 1]
        left_name = str(df.columns[i])
        right_name = str(df.columns[i + 1])

        if is_empty_col(left):
            merged = left.fillna("").astype(str) + right.fillna("").astype(str)
            new_name = f"{left_name} {right_name}"

            new_cols = []
            new_names = []

            for j in range(df.shape[1]):
                if j == i:
                    continue
                elif j == i + 1:
                    new_cols.append(merged)
                    new_names.append(new_name)
                else:
                    new_cols.append(df.iloc[:, j])
                    new_names.append(df.columns[j])

            df = pd.concat(new_cols, axis=1)
            df.columns = new_names
        else:
            i += 1

    return df

def strip_parentheses(text: str) -> str:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text

def extract_table_cells(doc:dict, text2table: bool, verbose:int = 0) -> dict:
    vals = {"cell": set(), "table": []}
    cells = []

    texts = []
    for i, t in enumerate(doc["texts"]):
        if i == 0:
            continue
        value = normalize_text(t.get("value"))
        title = normalize_text(t.get("title"))
        texts.append((title,value))
        if title:
            cells.append(title)
        else:
            if value:
                cells.append(value)

    if text2table:
        if texts != []:
            df = pd.DataFrame(texts)
            df = df.loc[:, ~((df == "") | df.isna()).all()]
            vals["table"].append(df)


    for table in doc["tables"]:
        header = []
        header_validated = False
        header_broken = []
        data = table["tableData"]
        table_rows = []
        table_rows_all = [] # for fallback
        for irow, row in enumerate(data):
            row_cleaned = [normalize_text(c).rstrip('.') for c in row]
            row_cleaned = [strip_parentheses(c) for c in row_cleaned]

            cells.extend(row_cleaned)
            if not header_validated:
                header = row_cleaned
                if is_header_broken(header):
                    header_broken.append(header)
                    header = []
                    header_validated = False
                    continue
            else:
                if "" in header:
                    if row_cleaned[0] == header[0]:
                        # double header
                        for i in range(len(header)):
                            if header[i] == "":
                                header[i] = row_cleaned[i]
                        continue
            if header_validated:
                # if row_cleaned are all None or empty, skip
                if all(c is None or c == "" for c in row_cleaned):
                    continue
                table_rows.append(row_cleaned)
            header_validated = True
            table_rows_all.append(row_cleaned)
        
        if table_rows == []:
            if verbose > 0:
                print(f"Document {doc['id']} has no valid table after header validation {table}.")
            vals["table"].append(merge_columns(pd.DataFrame(table_rows_all, index=None)))
        else:
            vals["table"].append(merge_columns(pd.DataFrame(table_rows, columns=header, index=None)))
            if verbose > 0 and header_broken != []:
                print(f"Broken headers {header_broken} are modified to {header} for document {doc['id']}")
        
        cells.extend(vals["table"][-1].values.flatten(order='F').tolist())
    
    cell_set = set(cells)
    cell_set.discard("")
    cell_set.discard("-")
    vals["cell"] = cell_set # sorted(cell_set, key=lambda x: (len(x), x), reverse=False)

    return vals


def remove_parentheses(text: str) -> str:
    # () とその中身を削除
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.strip()

def remove_numbers(text: str) -> str:
    # 数値っぽい文字だけならそのまま
    if re.fullmatch(r"[\d,.\-%\s]+", text):
        return text.strip()
    
    return re.sub(r"\d+", "", text).strip()


def divide_and_string(text:str) -> list[str]:
    # A & B -> [A, B]
    parts = re.split(r"\s*[&/]\s*", text)
    return [p.strip() for p in parts if p.strip()]

def divide_commma_string(text: str) -> list[str]:
    # A, B -> [A, B]
    # A: B -> [A, B]
    # A; B -> [A, B]
    parts = re.split(r"\s*[:,;]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def dedup_candidates(cands:list[str]) -> dict[str, list[str]]:
    counter = Counter()
    rep = {}
    first_idx = {}

    for i, c in enumerate(cands):
        k = soft_norm(c)
        if not k:
            continue
        counter[k] += 1
        if k not in rep:
            rep[k] = c
            first_idx[k] = i

    def sort_key(k):
        return (-counter[k], first_idx[k])

    candidate_keys = defaultdict(list)

    for k in counter.keys():
        candidate_keys[detect_value_type(rep[k])].append(k)
 
    candidate_list = defaultdict(list)
    for k in candidate_keys:
        # それぞれ頻度順にソート
        candidate_keys[k] = sorted(candidate_keys[k], key=sort_key)
        # 実際の文字列に戻す
        candidate_list[k] = [rep[k] for k in candidate_keys[k]]

    return candidate_list

def build_candidates(doc, table_cells: dict, strict: bool, substring_append: bool, value_append: bool,
                     remove_substrings: bool = False) -> dict[str, list[str]]:
    cands = []

    title = normalize_text(doc.get("title", ""))
    if title:
        cands.append(title)

    # 候補を増やす
    for cell in table_cells["cell"]:
        cands.extend(expand_table_cell(cell, substring_append=substring_append))
    
    if value_append:
        # 候補を増やす
        #cands.extend(extract_capitalized_phrases_ascii(doc))
        cands.extend(extract_capitalized_phrases_unicode(doc))

        cands.extend(extract_camel_case(doc))

        cands.extend(extract_single_token_proper_nouns(doc))

        cands.extend(extract_lowercase_noun_phrases(doc))

    if substring_append:
        cands.extend(expand_person_tokens(cands))

        cands.extend(expand_company_variants(cands))

        cands.extend([sub for c in cands for sub in divide_and_string(c)])
        cands.extend([sub for c in cands for sub in divide_commma_string(c)])

    cands.extend([remove_parentheses(c) for c in cands])
    cands.extend([remove_numbers(c) for c in cands])

    # strict=False: positive recall prioritization
    # strict=True : cleaner candidate pool for negative sampling
    filtered = []
    for c in cands:
        if strict:
            if not is_valid_negative_candidate(c, allow_number=True):
                continue
        else:
            if not is_valid_positive_candidate(c, allow_number=True):
                continue
        filtered.append(c)
    cands = filtered

    def _remove_substrings(candidates):
        candidates = sorted(set(candidates), key=len, reverse=True)
        result = []

        for c in candidates:
            pattern = r'\b' + re.escape(c) + r'\b'
            if not any(re.search(pattern, r) for r in result):
                result.append(c)

        return result
    
    if remove_substrings:
        cands = _remove_substrings(cands)

    return dedup_candidates(cands)



