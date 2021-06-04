for m in 40960 204800 1024000 5120000 10240000
do
    sbatch --export=TRAIN_DATA_SIZE=$m scripts/our_scripts/finetuning/finetune_dataset_size_toxicity_experts.sh
done