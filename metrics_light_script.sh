#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=800G
#SBATCH --time=18:00:00
#SBATCH --job-name=id
#SBATCH --output=output_job/id_job_%j.out
cd /u/dssc/zenocosini/helm_suite/MCQA_Benchmark
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate mcqa
export PYTHONPATH=/u/dssc/zenocosini/helm_suite/MCQA_Benchmark
#python probe/linear_probe_log.py --label letter
#python probe/linear_probe_log.py --label subject
python metrics_computer_light.py --conf-path config/config_small_metrics_1.json

#python label_overlap_computer.py
#python id_computer.py
echo "Running job: metrics"
