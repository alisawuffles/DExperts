PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=generations/sentiment/neutral_prompts/dapt/positive

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model models/dapt/finetuned_gpt2_positivity_gte99  \
    --p 0.9 \
    $OUTPUT_DIR
