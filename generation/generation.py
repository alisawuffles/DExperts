# TODO: add `text` key to cached generations
# TODO: consolidate code for loading cache
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Iterable, List

import openai
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

from generation.gpt2_generation import GPT2Generation
from generation.dexperts_generation import DExpertsGeneration
from generation.dexperts_gpt3_generation import DExpertsGPT3Generation
from generation.pplm_generation import PPLMGeneration
from utils.constants import OPENAI_API_KEY
from utils.utils import batchify, load_cache

logging.disable(logging.CRITICAL)  # Disable logging from transformers


def pplm(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         p: float,
         stepsize: float,
         batch_size: int,
         class_label: int,
         num_iterations: int,
         model_name_or_path: str,
         out_file: Path):
    # Set up PPLM with multiprocessing
    generator = PPLMGeneration(model_name_or_path, device=0)
    ctx = mp.get_context('spawn')
    generator.model.share_memory()
    generator.classifier.share_memory()
    pplm_func = partial(generator.__call__, class_label=class_label, stepsize=stepsize, num_iterations=num_iterations, length=max_len, top_p=p)

    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    with ctx.Pool(processes=batch_size) as pool:
        for batch in tqdm(pool.imap(pplm_func, prompts), total=len(prompts), desc='Generation', dynamic_ncols=True):
            for generation in batch:
                with out_file.open('a') as f:
                    print(json.dumps(generation), file=f)
                yield generation


def _pipeline_helper(prompts: pd.Series,
                     model_name_or_path: str,
                     max_len: int,
                     num_samples: int,
                     out_file: Path,
                     p: float,
                     **generate_kwargs):
    # Load cached generations
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    assert num_cached_generations % num_samples == 0

    # Remove prompts that have already been generated with
    prompts = prompts[num_cached_generations // num_samples:]
    if prompts.empty:
        return

    # Setup model
    generator = pipeline('text-generation', model=model_name_or_path, device=0)
    print("Created pipeline with model:", generator.model.__class__.__name__)

    # Generate with prompts
    for prompt in tqdm(prompts, desc='Generation', dynamic_ncols=True):
        # Generate
        # FIXME: this is a hack
        ctx_len = len(generator.tokenizer.tokenize(prompt))
        try:
            batch = generator(prompt,
                              num_return_sequences=num_samples,
                              clean_up_tokenization_spaces=True,
                              do_sample=True,
                              top_p=p,
                              max_length=ctx_len + max_len,
                              return_prompt=False,
                              **generate_kwargs)
            batch = map(lambda g: g['generated_text'][len(prompt):], batch)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and continuing...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * num_samples

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def ctrl(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         ctrl_code: str,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Prepend CTRL code to prompts
    prompts = ctrl_code + " " + prompts
    print(prompts)

    yield from _pipeline_helper(prompts=prompts,
                                model_name_or_path=model_name_or_path,
                                max_len=max_len,
                                num_samples=num_samples,
                                out_file=out_file,
                                **generate_kwargs)


def _gpt2_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generation,
                 out_file: Path,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size}):
        # Generate
        batch = generator.generate(prompt, max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2Generation(model_name_or_path)

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)


def dexperts(prompts: pd.Series,
             max_len: int,
             num_samples: int,
             batch_size: int,
             model_name_or_path: str,
             expert_name_or_path: str,
             antiexpert_name_or_path: str,
             out_file: Path,
             **generate_kwargs) -> Iterable[str]:

    generator = DExpertsGeneration(
        base_model=model_name_or_path, 
        expert_model=expert_name_or_path,
        antiexpert_model=antiexpert_name_or_path
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs
    )


def dexperts_gpt3(prompts: pd.Series,
                  max_len: int,
                  num_samples: int,
                  batch_size: int,
                  model_name_or_path: str,
                  expert_name_or_path: str,
                  antiexpert_name_or_path: str,
                  out_file: Path,
                  **generate_kwargs) -> Iterable[str]:
    
    generator = DExpertsGPT3Generation(
        gpt3_model=model_name_or_path,
        expert_model=expert_name_or_path,
        antiexpert_model=antiexpert_name_or_path
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs
    )


def gpt3(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         p: float,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path) -> Iterable[str]:
    openai.api_key = OPENAI_API_KEY

    def request(prompts: List[str]):
        # Retry request (handles connection errors, timeouts, and overloaded API)
        while True:
            try:
                return openai.Completion.create(
                    engine=model_name_or_path,
                    prompt=prompts,
                    max_tokens=max_len,
                    top_p=p,
                    n=1
                )
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                
    prompts = prompts.repeat(num_samples)
    for batch in tqdm(batchify(prompts, batch_size)):
        response = request(batch)
        yield from [choice['text'] for choice in response['choices']]