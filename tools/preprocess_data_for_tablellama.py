import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def table_to_string(data_dir):
    with open(os.path.join(data_dir, 'tables.json'), 'r') as f:
        tables = json.load(f)


    ## "input": "[TLE] The table caption is about tony lema. [TAB] | tournament | wins | top - 5 | top - 10 | top - 25 | events | cuts made [SEP] | masters tournament | 0 | 1 | 2 | 4 | 4 | 4 | [SEP] | us open | 0 | 2 | 3 | 4 | 6 | 5 | [SEP] | the open championship | 1 | 2 | 2 | 2 | 3 | 3 | [SEP] | pga championship | 0 | 0 | 1 | 2 | 5 | 4 | [SEP] | totals | 1 | 5 | 8 | 12 | 18 | 16 |",

    tables_string = {}
    for table in tables:
        tables_string[table['id']] = "[TLE] " + table['title'] + " [TAB] | " + " | ".join(table['header']) + " [SEP] | " + " [SEP] | ".join([" | ".join(row) for row in table['rows']]) + " |"

    return tables_string


def text_to_dict(data_dir):
    with open(os.path.join(data_dir, 'texts.json'), 'r') as f:
        texts = json.load(f)

    texts_dict = {}
    for text in texts:
        texts_dict[text['text_id']] = text['text'].replace("\n", " ")

    return texts_dict



if __name__ == "__main__":
    from string import Template

    data_dir = 'data'
    tables = table_to_string(data_dir)
    texts = text_to_dict(data_dir)

    instruction = "This is a releation judgement task. The goal of this task is to distinguish whether the given statement is related or unrelated to the given table."
    question = Template("The statement is:  <${statement}>. Is it related or unrelated to the table above?")

    df = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    train_data = []
    
    idx = 0
    for text_id, table_id, label in zip(df['text_id'], df['table_id'], df['label']):
        train_data.append({
            "idx": idx,
            "text_id": text_id,
            "table_id": table_id,
            "instruction": instruction,
            "input_seg": tables[table_id],
            "question": question.substitute(statement=texts[text_id][0:-1].replace("=","").replace("  "," ").strip()),
            "output": "related" if label == 1 else "unrelated"
        })
        idx += 1

    with open(os.path.join(data_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
