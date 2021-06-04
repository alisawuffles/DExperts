"""
construct sentiment data from SST-5 used to finetune sentiment (anti-)experts in DExperts
"""

import pytreebank

positive, negative = [],[]
dataset = pytreebank.load_sst()
examples = dataset['train'] + dataset['dev'] + dataset['test']

for ex in examples:
    label, sentence = ex.to_labeled_lines()[0]
    if label in [0,1]:      # negative or very negative
        negative.append(sentence)
    elif label in [3,4]:    # positive or very positive
        positive.append(sentence)
    elif label == 2:    # neutral
        pass
    else:
        raise ValueError

print(len(negative), len(positive))

with open('datasets/SST-5/negative.txt', 'w') as f:
    for l in negative:
        f.write(l + '\n')

with open('datasets/SST-5/positive.txt', 'w') as f:
    for l in positive:
        f.write(l + '\n')