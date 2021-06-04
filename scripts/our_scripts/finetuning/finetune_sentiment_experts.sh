DATA_DIR=datasets/SST-5/
BATCH_SIZE=4
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=16
SENTIMENT=negative

python -m scripts.finetuning.finetune_gpt2 \
	--output_dir models/experts/sentiment/large/finetuned_gpt2_$SENTIMENT \
	--model_type gpt2 \
	--model_name_or_path gpt2-large \
	--do_train \
	--num_train_epochs 3 \
	--block_size $BLOCK_SIZE \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/$SENTIMENT.txt \
	--overwrite_cache
