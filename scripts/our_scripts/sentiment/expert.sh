P=0.9
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=generations/sentiment/neutral_prompts/expert/positive

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model models/experts/sentiment/large/finetuned_gpt2_positive \
    --p $P \
    $OUTPUT_DIR
