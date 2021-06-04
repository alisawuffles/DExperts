ALPHA=2.0
API_RATE=20
expert_size=large
MODEL_DIR=models/experts/toxicity/$EXPERT_SIZE
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
OUTPUT_DIR=generations/toxicity/dexperts_gpt3/${EXPERT_SIZE}_experts/

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts-gpt3 \
    --model ada \
    --nontoxic-model $MODEL_DIR/finetuned_gpt2_nontoxic \
    --toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    --perspective-rate-limit $API_RATE \
    --batch-size 20 \
    --alpha $ALPHA \
    --filter_p 0.9 \
    --resume \
    $OUTPUT_DIR
