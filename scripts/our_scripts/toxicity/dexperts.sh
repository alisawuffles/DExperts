ALPHA=2.0
EXPERT_SIZE=large
API_RATE=20
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
OUTPUT_DIR=generations/toxicity/dexperts/${EXPERT_SIZE}_experts/

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    --perspective-rate-limit $API_RATE \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
