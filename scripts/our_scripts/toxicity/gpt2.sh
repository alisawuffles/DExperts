API_RATE=20
OUTPUT_DIR=generations/toxicity/gpt2

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file prompts/nontoxic_prompts-10k.jsonl \
    --model-type gpt2 \
    --model gpt2-large \
    --perspective-rate-limit $API_RATE \
    --p 0.9 \
    $OUTPUT_DIR
