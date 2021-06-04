import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
import click
import pandas as pd
import torch
from tqdm import tqdm
import json
import os
from transformers import pipeline
from generation.generation import gpt2, gpt3, ctrl, pplm, dexperts
from utils.utils import load_jsonl, batchify, ensure_dir

ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'pplm', 'ctrl']


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        yield {'text': generation, **response}


def collate(dataset: pd.DataFrame, generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    assert len(generations) % len(dataset) == 0
    n = len(generations) // len(dataset)
    print(f"Detected samples per prompt:", n)
    generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
    dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True, type=click.Choice(ALLOWED_MODELS))
@click.option('--pos-model', type=str, help="Positive model for DExperts")
@click.option('--neg-model', type=str, help="Negative model for DExperts")
@click.option('--positive/--negative', default=True, help="Sentiment for model_type='ctrl' or 'pplm'")
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--resume/--no-resume', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for ensemble methods')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus (top-p) sampling')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, model_type: str, 
         pos_model: str, neg_model: str, positive: bool, n: int, max_tokens: int, batch_size: int, resume: bool,
         alpha: float, p: float, filter_p: float):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']
    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ['gpt2', 'gpt2-ensemble', 'pplm']:
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'gpt2-ctrl':
            prompts = pd.Series('<|nontoxic|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / 'generations.jsonl'
    sentiment_file = output_dir / 'sentiment.jsonl'
    assert resume or not os.path.exists(generations_file)
    ensure_dir(output_dir)
    output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'

    # Setup model for generation
    if model_type == 'gpt2':
        generations_iter = gpt2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'gpt3':
        generations_iter = gpt3(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'dexperts':
        generations_iter = dexperts(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=pos_model,
            antiexpert_name_or_path=neg_model,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            alpha=alpha,
        )
    elif model_type == 'ctrl':
        assert model == 'ctrl'
        ctrl_code = "Reviews Rating 5.0" if positive else "Reviews Rating 1.0"
        generations_iter = ctrl(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            model_name_or_path=model,
            out_file=generations_file,
            p=p,
            # CTRL
            ctrl_code=ctrl_code,
            temperature=1.0,
            repetition_penalty=1.2
        )
    elif model_type == 'pplm':
        class_label = 2 if positive else 3
        generations_iter = pplm(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            class_label=class_label,
            stepsize=0.20,
            num_iterations=10,
            model_name_or_path=model,
            out_file=generations_file
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # read generations
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)
    assert len(generations) % len(prompts) == 0
    n = len(generations) // len(prompts)

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis')
    with open(sentiment_file, 'w') as fo:
        for i, p in tqdm(enumerate(prompts), total=len(prompts), desc='Scoring generations'):
            sentences_for_prompt = []
            for j in range(n):
                gen = generations[i*n + j]
                sentences_for_prompt.append(f'{p}{gen}')
            try:
                predictions_for_prompt = classifier(sentences_for_prompt)
            except IndexError: # sometimes the generation is too long?
                predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
            for res in predictions_for_prompt:
                fo.write(json.dumps(res) + '\n')

    torch.cuda.empty_cache()
    print('Finished generation and sentiment scoring!')

    if os.path.exists(sentiment_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(sentiment_file), output_file)


if __name__ == '__main__':
    main()
