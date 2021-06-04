"""
construct toxicity data from OWT which is used to further pretrain GPT2 in DAPT baseline
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scripts.data.openwebtext import OWTC


DATA_DIRECTORY = Path("/gscratch/scrubbed/ahai/messy/domain_data/")
OUT_DIRECTORY = Path("datasets/openwebtext")
path_to_openwebtext_corpus = DATA_DIRECTORY / "openwebtext_shards"
path_to_metadata = OUT_DIRECTORY / 'openwebtext_toxicity_meta.jsonl'
attributes = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat', 'profanity', 'sexually_explicit', 'flirtation']

owtc = OWTC(path_to_corpus=path_to_openwebtext_corpus)
owtc.load_corpus(batch_size=1000)
metadata_df = pd.read_json(path_to_metadata, lines=True)

for a in attributes:
    metadata_df[a] = None

for i, row in tqdm(metadata_df.iterrows()):
    for a in attributes:
        metadata_df.at[i, a] = row['perspective_doc_score'][a]

# merge OWT with its toxicity metadata!
owtc_metadata_df = owtc.corpus.merge(metadata_df, on='md5_hash')

# create pretraining data
fos = defaultdict(dict)
percentiles = defaultdict(dict)
for a in attributes:
    fos[a][0.99] = open(OUT_DIRECTORY / f'{a}_gte99.txt', 'w')
    fos[a][0.02] = open(OUT_DIRECTORY / f'{a}_lte2.txt', 'w')
    percentiles[a][0.99] = owtc_metadata_df[a].quantile(0.99)
    percentiles[a][0.02] = owtc_metadata_df[a].quantile(0.02)

for i, row in tqdm(owtc_metadata_df.iterrows()):
    for a in attributes:
        if row[a] is None:
            continue
        if row[a] >= percentiles[a][0.99]:
            fos[a][0.99].write(row['text'])
        if row[a] <= percentiles[a][0.02]:
            fos[a][0.02].write(row['text'])

for a in attributes:
    fos[a][0.99].close()
    fos[a][0.02].close()

    