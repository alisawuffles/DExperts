echo "TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE"

DATA_DIR=datasets/jigsaw-unintended-bias-in-toxicity-classification/
SCRUBBED_DIR=/gscratch/scrubbed/alisaliu/
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16
BLOCK_SIZE=128
CHECKPOINTS_PER_EPOCH=5
TRAIN_EPOCHS=2
SAVE_STEPS=$(($TRAIN_DATA_SIZE / ($BATCH_SIZE * $GRAD_ACCUM_STEPS * $BLOCK_SIZE * $CHECKPOINTS_PER_EPOCH)))
echo "SAVE_STEPS=$SAVE_STEPS"

python -m scripts.finetuning.finetune_gpt2_dataset_size \
	--output_dir $SCRUBBED_DIR/models/finetuned_gpt2_nontoxic_${TRAIN_DATA_SIZE} \
	--model_type gpt2 \
	--model_name_or_path gpt2-large \
	--do_train \
	--num_train_epochs $TRAIN_EPOCHS \
	--block_size $BLOCK_SIZE \
	--train_data_size $TRAIN_DATA_SIZE \
	--save_steps $SAVE_STEPS \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/toxicity_eq0.txt \
	--overwrite_cache
