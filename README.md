# DExperts
Hi! This repository contains code for the paper [DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts](https://arxiv.org/abs/2105.03023) to appear at ACL 2021. If you have any questions, please feel free to create a Github issue or reach out to the first author at alisaliu@cs.washington.edu. 

Create a conda environment called `dexperts` with
```
conda env create -f environment.yml
```

## Toxicity
To generate continuations with DExperts and score them for toxicity using the [PerspectiveAPI](https://github.com/conversationai/perspectiveapi) toxicity scorer, run the following command.
```
OUTPUT_DIR=generations/toxicity/dexperts
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    --perspective-rate-limit $API_RATE \
    --alpha 2.0 \
    --filter_p 0.9 \
    $OUTPUT_DIR
```

In general, `model_type` is one of `gpt2` (the base model), `dexperts` (our method), and `pplm`. With an [OpenAI API](https://beta.openai.com/) key for GPT-3 access, you can also try `gpt3` and `dexperts-gpt3`. Different methods have different additional parameters to specify; to see the commands we used for each method in our paper, please look under `scripts/our_scripts/toxicity`. For experiments with GeDi, we directly used the original [authors' codebase](https://github.com/salesforce/GeDi). 

When `model_type` is `dexperts`, we can steer away from toxicity using only a toxic anti-expert. To do this, leave `--nontoxic-model` empty, and DExperts will re-use the base model as the expert. The hyperparameter `alpha` controls the strength of steering over the base model. We use `filter_p` to use the nucleus from the base model, as described in Section 2.2 of our paper.

This script will create three files in `OUTPUT_DIR`: `generations.jsonl` with all of the generated continuations, `perspective.jsonl` with all the scores from Perspective API, and `prompted_gens_[model_type].jsonl`, which collates the previous two files.

To try a model's output on your own prompts, simply create your own prompts file! To see the format of the prompts file, see `prompts/toy_prompt.jsonl`.

## Sentiment
To generate continuations with DExperts conditioned on sentiment prompts and score them for sentiment using HuggingFace's sentiment classifier, run the following command.

```
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=generations/sentiment/neutral_prompts/dexperts/positive/

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/finetuned_gpt2_positive \
    --neg-model $MODEL_DIR/finetuned_gpt2_negative \
    --alpha 3.2 \
    --filter_p 0.9 \
    $OUTPUT_DIR
```

The `model_type` can be any of the options from before, with the addition of `ctrl`. Again, the full commands used for each method can be found under `scripts/our_scripts/sentiment`.

When `model_type` is `dexperts`, we always interpret `--pos-model` as the expert and `--neg-model` as the anti-expert; for negative steering, use `alpha` < 0. By leaving one of `--pos-model` or `--neg-model` empty, DExperts will re-use the base model as the missing expert or anti-expert.

## Evaluation
To evaluate generated output for fluency and diversity, run the following command. The `GENERATIONS_FILE` should have the format `prompted_gens_[model_type].jsonl`.
```
python -m scripts.evaluation.evaluate_generations \
    --generations_file $GENERATIONS_FILE
```

## Notebooks
Our jupyter notebooks are in `notebooks/`. To obtain the same tables and plots that appear in the paper, look in `sentiment_results.ipynb`, `toxicity_results.ipynb`, and `human_eval_results.ipynb`. To create your own prompts dataset with a couple lines of code, you can get started with `prompts_playground.ipynb`. Sample and compare generations from each model with `review_sentiment_generations.ipynb` and `review_toxicity_generations.ipynb`. 

## Downloading the original data and models from our paper

To download the prompts we used for evaluation, generations output by each model, and finetuning datasets from our paper, ensure you have `gdown` installed, then run the following commands inside the `dexperts/` root directory. Descriptions of the contents of each of these folders can be found within the folder.
```
# prompts
gdown https://drive.google.com/uc?id=1bI49aJvmEoLdqSNb30JkORdsNJmv7Aep
unzip prompts.zip && rm prompts.zip
# generations
gdown https://drive.google.com/uc?id=10jL1-eCv8w3oeGFgA_jrel0enrNVdFW7
unzip generations.zip && rm generations.zip
# datasets
gdown https://drive.google.com/uc?id=1MeEjLPxQ77AYtzL0nd1hYJTlL8OJgHkI
unzip datasets.zip && rm datasets.zip
```

To download models from our paper,
```
mkdir models
cd models
# (anti-)expert models
gdown https://drive.google.com/uc?id=1HSrNMrq4OZ3nyTobNd2TZFcB5NYwluu-
unzip experts.zip && rm experts.zip
# DAPT models
gdown https://drive.google.com/uc?id=1eDlRU04s-H1elWWtPuDoBNAqyoqj3_p9
unzip dapt.zip && rm dapt.zip
# PPLM classifiers
gdown https://drive.google.com/uc?id=17s26QM9vJp9hCUkRBrDx5Wa__4BlrqGL
unzip pplm_classifiers.zip && rm pplm_classifiers.zip
```

## Citation
```
@inproceedings{liu-etal-2021-dexperts,
    title = "{DExperts}: Decoding-Time Controlled Text Generation with Experts and Anti-Experts",
    author = "Alisa Liu and Maarten Sap and Ximing Lu and Swabha Swayamdipta and Chandra Bhagavatula and Noah A. Smith and Yejin Choi",
    booktitle = "Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)",
    year = "2021",
    url = "https://arxiv.org/abs/2105.03023",
}
```

This code was built on top of [allenai/real-toxicity-prompts](https://github.com/allenai/real-toxicity-prompts) and with inspiration from [yangkevin2/naacl-2021-fudge-controlled-generation](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation).