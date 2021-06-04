"""
preprocessing script to construct training data for PPLM toxicity classifier
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('datasets/jigsaw-toxic-comment-classification-challenge/')
toxicity_df = pd.read_csv(DATA_DIR / 'train.csv')

with open(DATA_DIR / 'toxic_train.txt', 'wb') as fo:
    for i, row in tqdm(toxicity_df.iterrows()):
        comment_text = row["comment_text"].encode()
        ex = f'{{"text": {comment_text}, "label": [{row["toxic"]}, {row["severe_toxic"]}, {row["obscene"]}, {row["threat"]}, {row["insult"]}, {row["identity_hate"]}]}}\n'.encode()
        fo.write(ex)
