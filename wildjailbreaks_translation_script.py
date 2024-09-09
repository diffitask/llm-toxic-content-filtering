import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
from time import time
import json

from datasets import load_dataset

# Assuming you have a DataFrame `train_df`
with open('translated_sentences.json', 'w', encoding='utf-8') as f:
    pass

# train_df = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
train_df = pd.read_csv("sampled_wildjailbreaks.csv").fillna("")
        
sentences = train_df["vanilla"]  # All sentences in the column
translated_sentences = []

batch_size = 100
start_time = time()

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]
    batch_translations = []
    
    for sentence in tqdm(batch):
        if sentence == '':
            batch_translations.append({
                'original': '',
                'translated': ''
            })
            continue
    
        translated_sentence = GoogleTranslator(source='en', target='ru').translate(sentence)
        batch_translations.append({
            'original': sentence,
            'translated': translated_sentence
        })
    
    translated_sentences.extend(batch_translations)

    # Save each batch to a JSON file
    with open('translated_sentences.json', 'a', encoding='utf-8') as f:
        json.dump(batch_translations, f, ensure_ascii=False, indent=4)
        f.write('\n')  # Add newline after each batch to maintain formatting
    
    print(f"Batch {i // batch_size + 1} translated and saved.")

total_time = time() - start_time
print(f"Total translation time: {total_time} seconds")