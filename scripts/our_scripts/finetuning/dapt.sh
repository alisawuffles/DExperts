DATA_DIR=datasets/openwebtext/
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16
BLOCK_SIZE=128

python -m scripts.finetuning.finetune_gpt2 \
    --model_type gpt2 \
    --model_name_or_path gpt2-large \
    --do_train \
    --num_train_epochs 3 \
    --block_size $BLOCK_SIZE \
    --save_total_limit 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --train_data_file $DATA_DIR/negativity_gte99.txt \
    --output_dir models/dapt/finetuned_gpt2_negativity_gte99 \
    --overwrite_output_dir
