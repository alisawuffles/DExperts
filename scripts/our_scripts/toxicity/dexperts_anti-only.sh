ALPHA=2.0
API_RATE=20
EXPERT_SIZE=large
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
OUTPUT_DIR=generations/toxicity/dexperts_anti-only

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    --perspective-rate-limit $API_RATE \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
