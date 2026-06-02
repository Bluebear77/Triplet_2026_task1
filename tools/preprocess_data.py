import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def table_to_csv(data_dir):
    with open(os.path.join(data_dir, 'tables.json'), 'r') as f:
        tables = json.load(f)

    titles = {}

    os.makedirs(os.path.join(data_dir, 'tables'), exist_ok=True)

    for table in tables:
        id = table['id']
        title = table['title']
        titles[id] = title    

        df = pd.DataFrame(table['rows'], columns=table['header'])
        df.to_csv(os.path.join(data_dir, 'tables', f'{id}.csv'), index=False, encoding='utf8')


    with open(os.path.join(data_dir, 'titles.json'), 'w') as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)

def pairs_to_map(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    text_to_table = {}
    table_to_text = {}

    for text_id, table_id, label in zip(df['text_id'], df['table_id'], df['label']):
        if text_id not in text_to_table:
            text_to_table[text_id] = {'positive': [], 'negative': []}
        if label == 1:
            text_to_table[text_id]['positive'].append(table_id)
        else:
            text_to_table[text_id]['negative'].append(table_id)

        if table_id not in table_to_text:
            table_to_text[table_id] = {'positive': [], 'negative': []}
        if label == 1:
            table_to_text[table_id]['positive'].append(text_id)
        else:
            table_to_text[table_id]['negative'].append(text_id)

    with open(os.path.join(data_dir, 'text_to_table.json'), 'w') as f:
        json.dump(text_to_table, f, indent=2)
    with open(os.path.join(data_dir, 'table_to_text.json'), 'w') as f:
        json.dump(table_to_text, f, indent=2)

    print('len(text):', len(text_to_table))

    count_positive_negative_from_pair_map(text_to_table, entity_type='text')

    print('len(table):', len(table_to_text))
    count_positive_negative_from_pair_map(table_to_text, entity_type='table')

def count_positive_negative(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    positive_count = (df['label'] == 1).sum()
    negative_count = (df['label'] == 0).sum()
    print(f'Positive pairs: {positive_count}')
    print(f'Negative pairs: {negative_count}')

def count_positive_negative_from_pair_map(pair_map, entity_type='text'):
    positives = {}
    negatives = {}
    for k, v in pair_map.items():
        positive_count = len(v['positive'])
        positives[positive_count] = positives.get(positive_count, 0) + 1
        negative_count = len(v['negative'])
        negatives[negative_count] = negatives.get(negative_count, 0) + 1
    for count, num in sorted(positives.items()):
        print(f'{count} positive examples: {num} {entity_type}s')
    
    for count, num in sorted(negatives.items()):
        print(f'{count} negative examples: {num} {entity_type}s')
    
    plt.figure(figsize=(7.2, 3))
    plt.bar(positives.keys(), positives.values(), label='Positive', alpha=0.5)
    plt.bar(negatives.keys(), negatives.values(), label='Negative', alpha=0.5)
    plt.xlabel(f'Number of {entity_type} examples')
    plt.ylabel('Number of entities')
    plt.xlim(-0.5, 20)
    plt.xticks(range(0, 21, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'{entity_type}_examples_distribution.png'))
    plt.close()
    


if __name__ == "__main__":
    
    data_dir = 'data'
    table_to_csv(data_dir)
    count_positive_negative(data_dir)
    pairs_to_map(data_dir)
