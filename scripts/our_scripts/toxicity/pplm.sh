API_RATE=20
OUTPUT_DIR=generations/toxicity/pplm
PROMPTS_DATASET=nontoxic_prompts-10k/by_1k/prompts_0.jsonl

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type pplm \
	--model 'toxicity-large' \
    --p 0.9 \
    --batch-size 1 \
    --perspective-rate-limit $API_RATE \
	$OUTPUT_DIR
