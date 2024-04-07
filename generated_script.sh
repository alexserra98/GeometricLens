#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=230G
#SBATCH --time=18:00:00
#SBATCH --job-name=inference
#SBATCH --output=output_job/inference_job_%j.out
cd /u/dssc/zenocosini/helm_suite/MCQA_Benchmark
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate mcqa
export PYTHONPATH=/u/dssc/zenocosini/helm_suite/MCQA_Benchmark
export CUDA_VISIBLE_DEVICES=0,1,2,3
python inference.py --conf-path config/config_commonsense_wrong.json
echo "Running job: inference"
