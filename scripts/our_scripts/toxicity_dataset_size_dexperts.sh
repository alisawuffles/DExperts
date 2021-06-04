block_size=128
effective_batch_size=64
checkpoints_per_epoch=5

for m in 40960 5120000 204800 1024000 10240000
do
    for i in $(seq 1 5)
    do
        c=$((i * m / (effective_batch_size * block_size * checkpoints_per_epoch)))
        id=$(sbatch --parsable --export=CHECKPOINT=$c,TRAIN_DATA_SIZE=$m scripts/our_scripts/toxicity/dataset_size_dexperts.sh)
        echo "Submitted batch job $id"
        id=$(sbatch --parsable --dependency=afterany:$id --export=CHECKPOINT=$c,TRAIN_DATA_SIZE=$m scripts/our_scripts/toxicity/score/score_dataset_size_dexperts.sh)
        echo "Submitted batch job $id"
        sbatch --dependency=afterany:$id --export=generations_file=generations/toxicity/dataset_size_dexperts/m-$m/checkpoint-$c/prompted_gens_dexperts.jsonl scripts/our_scripts/eval/evaluate_generations.sh 
    done
done