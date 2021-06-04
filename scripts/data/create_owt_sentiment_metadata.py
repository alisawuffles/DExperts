"""
score every document in OWT for sentiment using HuggingFace's sentiment-analysis pipeline,
output metadata files
"""

import click
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from scripts.data.openwebtext import OWTC
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers import pipeline
import time

DATA_DIRECTORY = Path("/gscratch/scrubbed/ahai/messy/domain_data/")
OUT_DIRECTORY = Path("datasets/openwebtext")


@click.command()
@click.option('--shard', required=False, type=int)
def main(shard: int):
    print(f'--- Shard: {shard} ---')
    path_to_openwebtext_corpus = DATA_DIRECTORY / "openwebtext_shards"

    owtc = OWTC(path_to_corpus=path_to_openwebtext_corpus, shard=shard)
    owtc.load_corpus(batch_size=1000)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    classifier = pipeline('sentiment-analysis')
    max_seq_len = classifier.model.config.max_position_embeddings

    lines_per_flush = 10000
    with open(f'datasets/openwebtext/openwebtext_sentiment_meta_{shard}.jsonl', 'w') as fo:
        for i, row in tqdm(owtc.corpus.iterrows(), total=len(owtc.corpus.index), desc='Scoring OpenWebText'):
            last_flush = time.time()
            token_ids = tokenizer.encode_plus(
                row['text'], 
                add_special_tokens=False, 
                padding=True, 
                truncation=True,
                max_length=max_seq_len-2,   # the classifier will add 2 special tokens
                return_tensors='pt')['input_ids']
            truncated_text = tokenizer.decode(token_ids.squeeze(0))
            prediction = classifier(truncated_text)[0]
            row = row.drop(['text', 'url', 'subreddit', 'karma'])
            row_dict = row.to_dict()
            row_dict['sentiment_score'] = prediction
            fo.write(json.dumps(row_dict, default=str) + '\n')
            
            if i % lines_per_flush == 0:
                fo.flush()


if __name__ == '__main__':
    main()
