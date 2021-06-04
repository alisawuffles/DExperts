import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm
import os

from generation.generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.utils import load_jsonl, batchify, ensure_dir

ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm']


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if response['response']:
            response = unpack_scores(response['response'])[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
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
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--toxic-model', type=str, default=None, help='Anti-expert for DExperts')
@click.option('--nontoxic-model', type=str, default=None, help='Expert for DExperts')
@click.option('--perspective-rate-limit', default=25)
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--resume/--no-resume', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for dexperts')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, model_type: str, nontoxic_model: str,
         toxic_model: str, perspective_rate_limit: int, n: int, max_tokens: int, batch_size: int, resume: bool,
         alpha: float, filter_p: float, p: float):
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
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-ensemble', 'gpt2-naughty-list', 'pplm']:
            prompts = pd.Series('<|endoftext|>')
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
    perspective_file = output_dir / 'perspective.jsonl'
    assert resume or not os.path.exists(generations_file)   # don't overwrite generations!
    ensure_dir(output_dir)
    output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'

    # Create perspective worker thread
    perspective = PerspectiveWorker(
        out_file=perspective_file,
        total=len(prompts) * n,
        rate_limit=perspective_rate_limit
    )

    # Setup model for generation
    # TODO: move this logic into generation.py
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
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            alpha=alpha,
        )
    elif model_type == 'dexperts-gpt3':
        generations_iter = dexperts_gpt3(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            alpha=alpha,
        )
    elif model_type == 'pplm':
        generations_iter = pplm(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            class_label=0,
            stepsize=0.20,
            num_iterations=10,
            model_name_or_path=model,
            out_file=generations_file
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate and collate perspective scores
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)
        perspective(f'generation-{i}', gen)

    torch.cuda.empty_cache()
    perspective.stop()
    print('Finished generation and perspective scoring!')

    if os.path.exists(perspective_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()
