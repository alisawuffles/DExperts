PROMPTS_DATASET=prompts/sentiment_prompts-10k/positive_prompts/by_1k/prompts_0.jsonl
OUTPUT_DIR=generations/sentiment/neutral_prompts/pplm/positive

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type pplm \
	--model 'sentiment-large' \
    --positive \
    --p 0.9 \
    --batch-size 1 \
	$OUTPUT_DIR
