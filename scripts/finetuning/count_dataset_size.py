"""
script to count the size of the dataset in terms of tokens
"""

from scripts.finetuning.text_dataset import TextDataset
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path('datasets/SST-5')
file_path = DATA_DIR / f'positive.txt'
BLOCK_SIZE = 128

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

train_dataset = TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=BLOCK_SIZE, overwrite_cache=True)
print((len(train_dataset)-1) * BLOCK_SIZE + len(train_dataset.examples[-1]))