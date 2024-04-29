#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=metrics
#SBATCH --output=output_job/metrics_job_%j.out
cd /u/dssc/zenocosini/helm_suite/MCQA_Benchmark
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate mcqa
export PYTHONPATH=/u/dssc/zenocosini/helm_suite/MCQA_Benchmark
export CUDA_VISIBLE_DEVICES=0,1,2,3
python metrics_computer.py --conf-path config/config_metrics.json
echo "Running job: metrics"
