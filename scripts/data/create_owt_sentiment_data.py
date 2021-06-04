"""
construct sentiment data from OWT which is used to further pretrain GPT2 in DAPT baseline,
using metadata created by create_owt_sentiment_metadata.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json, jsonlines
from tqdm import tqdm
from collections import defaultdict
from scripts.data.openwebtext import OWTC


DATA_DIRECTORY = Path("/gscratch/scrubbed/ahai/messy/domain_data/")
OUT_DIRECTORY = Path("datasets/openwebtext")
path_to_openwebtext_corpus = DATA_DIRECTORY / "openwebtext_shards"
path_to_metadata = OUT_DIRECTORY / 'openwebtext_sentiment_meta.jsonl'

owtc = OWTC(path_to_corpus=path_to_openwebtext_corpus)
owtc.load_corpus(batch_size=1000)
metadata_df = pd.read_json(path_to_metadata, lines=True)

for col in ['label', 'score']:
    metadata_df[col] = None

for i, row in tqdm(metadata_df.iterrows()):
    metadata_df.at[i, 'label'] = row['sentiment_score']['label']
    metadata_df.at[i, 'score'] = row['sentiment_score']['score']

# merge OWT with its sentiment metadata!
owtc_metadata_df = owtc.corpus.merge(metadata_df, on='md5_hash')

pos_p99 = owtc_metadata_df.loc[owtc_metadata_df['label'] == 'POSITIVE']['score'].quantile(0.99)
neg_p99 = owtc_metadata_df.loc[owtc_metadata_df['label'] == 'NEGATIVE']['score'].quantile(0.99)

with open(OUT_DIRECTORY / 'positive_gte99.txt', 'w') as pos_fo, open(OUT_DIRECTORY / 'negative_gte99.txt', 'w') as neg_fo:
    for i, row in tqdm(owtc_metadata_df.iterrows()):
        if row['label'] == 'POSITIVE' and row['score'] > pos_p99:
            pos_fo.write(row['text'])
        elif row['label'] == 'NEGATIVE' and row['score'] > neg_p99:
            neg_fo.write(row['text'])