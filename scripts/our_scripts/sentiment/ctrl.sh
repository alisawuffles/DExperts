P=0.9
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=generations/prompted_sentiment-10k/neutral_prompts/ctrl/positive/

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type ctrl \
    --model ctrl \
    --positive \
    --p $P \
    $OUTPUT_DIR
