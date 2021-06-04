API_RATE=20
OUTPUT_DIR=generations/toxicity/dapt/

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file prompts/nontoxic_prompts-10k.jsonl \
    --model-type gpt2 \
    --model models/dapt/finetuned_gpt2_toxicity_lte2 \
    --perspective-rate-limit $API_RATE \
    --p 0.9 \
    $OUTPUT_DIR
