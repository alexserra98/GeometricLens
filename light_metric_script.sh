#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=680G
#SBATCH --time=2-20:00:00
#SBATCH --job-name=probe
#SBATCH --output=output_job/probe_job_%j.out
cd /u/dssc/zenocosini/helm_suite/MCQA_Benchmark
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate mcqa
export PYTHONPATH=/u/dssc/zenocosini/helm_suite/MCQA_Benchmark
python probe/linear_probe_log.py --label letter 
python probe/linear_probe_log.py --label subject 
echo "Running job: metrics"
