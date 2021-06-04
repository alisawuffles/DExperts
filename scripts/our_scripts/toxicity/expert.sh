API_RATE=20
OUTPUT_DIR=generations/toxicity/nontoxic_expert

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file prompts/nontoxic_prompts-10k.jsonl \
    --model-type gpt2 \
    --model models/experts/toxicity/large/finetuned_gpt2_nontoxic \
    --perspective-rate-limit $API_RATE \
    --p 0.9 \
    $OUTPUT_DIR
