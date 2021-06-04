"""
construct toxicity data from Jigsaw used to finetune nontoxic expert and toxic anti-expert in DExperts
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

data_dir = 'datasets/jigsaw-unintended-bias-in-toxicity-classification'
jigsaw_df = pd.read_csv(f'{data_dir}/all_data.csv')
attributes = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat', 'obscene', 'sexual_explicit']

fos = defaultdict(dict)
for a in attributes:
    fos[a]['toxic'] = open(f'{data_dir}/{a}_gte0.5.txt', 'w')
    fos[a]['nontoxic'] = open(f'{data_dir}/{a}_eq0.txt', 'w')

comments_ct = {a: {'gte50': 0, 'eq0': 0} for a in attributes}
for i, row in tqdm(jigsaw_df.iterrows(), total=len(jigsaw_df.index)):
    for a in attributes:
        if row[a] >= 0.5:
            fos[a]['toxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['gte50'] += 1
        if row[a] == 0.0:
            fos[a]['nontoxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['eq0'] += 1

for a in attributes:
    fos[a]['toxic'].close()
    fos[a]['nontoxic'].close()

print(comments_ct)
