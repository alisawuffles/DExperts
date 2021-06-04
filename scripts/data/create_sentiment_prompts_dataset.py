"""
create sentiment prompts dataset of size 100K from OWT
"""

import numpy as np
import pandas as pd
import json, jsonlines
from tqdm import tqdm
from scripts.data.openwebtext import OWTC
import nltk
from pathlib import Path
import sys, os

PROMPTS_DIR = Path('/gscratch/xlab/alisaliu/language-model-toxicity/prompts/')
DATA_DIRECTORY = Path("/gscratch/scrubbed/ahai/messy/domain_data/")
path_to_openwebtext_corpus = DATA_DIRECTORY / "openwebtext_shards"

owtc = OWTC(path_to_corpus=path_to_openwebtext_corpus)
owtc.load_corpus(batch_size=1000)

n = 100000
with open(PROMPTS_DIR / 'sentiment_prompts.jsonl', 'w') as fo:
    for i, doc in tqdm(owtc.corpus.sample(n).iterrows(), total=n):
        sentences = nltk.sent_tokenize(doc.text)
        for sentence in sentences:
            tokenized_text = nltk.word_tokenize(sentence)   # list of tokens
            # token-level index of split
            tok_idx = len(tokenized_text) // 2
            if not (4 <= tok_idx <= 10):
                continue
            # get character-level index of split in original sentence
            word_to_split_on = tokenized_text[tok_idx]
            tok_ct = np.sum([token.count(word_to_split_on) for token in tokenized_text[:tok_idx+1]])
            char_idx = sentence.replace(word_to_split_on, 'X'*len(word_to_split_on), tok_ct-1).find(word_to_split_on)
            # update prompts and conts
            prompt = sentence[:char_idx].strip()
            cont = sentence[char_idx:].strip()
            # write output
            l = {}
            l['md5_hash'] = doc['md5_hash']
            l['prompt'] = {'text': prompt}
            l['continuation'] = {'text': cont}
            fo.write(json.dumps(l, default='str') +'\n')
            