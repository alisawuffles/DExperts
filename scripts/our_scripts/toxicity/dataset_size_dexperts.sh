API_RATE=20
ALPHA=2.0
SCRUBBED_DIR=/gscratch/scrubbed/alisaliu
EXPERT=$SCRUBBED_DIR/models/finetuned_gpt2_nontoxic_$TRAIN_DATA_SIZE/checkpoint-$CHECKPOINT
ANTIEXPERT=$SCRUBBED_DIR/models/finetuned_gpt2_toxic_$TRAIN_DATA_SIZE/checkpoint-$CHECKPOINT
PROMPTS_DATASET=prompts/nontoxic_prompts-10k/by_1k/prompts_0.jsonl
OUTPUT_DIR=generations/toxicity/dataset_size_dexperts/m-$TRAIN_DATA_SIZE/checkpoint-$CHECKPOINT/

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --nontoxic-model $EXPERT \
    --toxic-model $ANTIEXPERT \
    --perspective-rate-limit $API_RATE \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
