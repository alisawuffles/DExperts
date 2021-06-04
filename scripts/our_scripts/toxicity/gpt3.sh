API_RATE=20
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
OUTPUT_DIR=generations/toxicity/gpt3/

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt3 \
    --model ada \
    --perspective-rate-limit $API_RATE \
    --batch-size 20 \
    --p 0.9 \
    $OUTPUT_DIR
