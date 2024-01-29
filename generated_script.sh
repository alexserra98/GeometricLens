#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=inference
#SBATCH --output=output_job/inference_job_%j.out
cd /u/dssc/zenocosini/helm_suite/inference_id
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate crfm-helm
export PYTHONPATH=/u/dssc/zenocosini/helm_suite/
export CUDA_VISIBLE_DEVICES=0,1,2,3
python inference.py --conf-path config.json
echo "Running job: inference"
