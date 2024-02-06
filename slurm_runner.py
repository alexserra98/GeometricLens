import argparse
import subprocess
import os
from collections import namedtuple

def create_bash_script(script_name, arguments, config_path):
    with open(script_name, 'w') as script_file:
        script_file.write('#!/bin/bash\n')
        # Add any Slurm SBATCH directives here
        script_file.write(f'#SBATCH --partition={arguments.partition}\n')
        script_file.write(f'#SBATCH --account=LADE\n')
        script_file.write(f'#SBATCH --nodes={arguments.nodes}\n')
        script_file.write(f'#SBATCH --exclusive\n')
        #script_file.write(f'#SBATCH --begin=now+4hour\n')
        script_file.write(f'#SBATCH --ntasks-per-node={arguments.ntasks_per_node}\n')
        script_file.write(f'#SBATCH --cpus-per-task={arguments.cpus_per_task}\n')
        script_file.write(f'#SBATCH --mem={arguments.mem}\n')
        script_file.write(f'#SBATCH --time={arguments.time}\n')
        script_file.write(f'#SBATCH --job-name={arguments.job_name}\n')
        script_file.write(f'#SBATCH --output=output_job/{arguments.output}_job_%j.out\n')
        script_file.write(f'cd /u/dssc/zenocosini/helm_suite/MCQA_Benchmark\n') 
        script_file.write(f'module load cuda/11.8\n')
        script_file.write(f'eval "$(conda shell.bash hook)"\n')
        script_file.write(f'conda activate mcqa\n')
        script_file.write(f'export PYTHONPATH=/u/dssc/zenocosini/helm_suite/MCQA_Benchmark\n')
        script_file.write(f'export CUDA_VISIBLE_DEVICES=0,1,2,3\n') 
        script_file.write(f'export OMP_NUM_THREADS=1\n') 
        if arguments.job_type == 'inference':
            script_file.write(f'python inference.py --conf-path {config_path}\n')
        elif arguments.job_type == 'metrics':
            script_file.write(f'python metrics_computer.py --conf-path {config_path}\n')

        # Example command to run
        script_file.write(f'echo "Running job: {arguments.job_name}"\n')
        # Add more commands based on your requirements



def main():
    parser = argparse.ArgumentParser(description="Create and submit a job to Slurm.")
    parser.add_argument('--job-type', type=str, help='Job name', required=True)
    parser.add_argument('--conf-path', type=str, help='Path to configuration file', required=False)
    args = parser.parse_args()
    
    job_type = args.job_type
    config_path = args.conf_path 
    
    SbatchArgs = namedtuple('SbatchArgs', [ 'job_type',
                                        'partition', 
                                        'nodes', 
                                        'ntasks_per_node', 
                                        'cpus_per_task', 
                                        'mem', 
                                        'time', 
                                        'job_name', 
                                        'output'])
    
    
    if job_type == 'inference':
        sbatch_args = SbatchArgs(job_type='inference',
                                 partition='GPU',
                                 nodes=2,
                                 ntasks_per_node=1,
                                 cpus_per_task=1,
                                 mem='230G',
                                 time='18:00:00',
                                 job_name='inference',
                                 output='inference')
    elif job_type == 'metrics':
        sbatch_args = SbatchArgs(job_type='metrics',
                                 partition='GPU',
                                 nodes=1,
                                 ntasks_per_node=1,
                                 cpus_per_task=1,
                                 mem='230G',
                                 time='1-20:00:00',
                                 job_name='metrics',
                                 output='metrics')    
    

    script_name = 'generated_script.sh'
    create_bash_script(script_name, sbatch_args,config_path)

    # Submit the script to Slurm
    subprocess.run(['sbatch', script_name])

if __name__ == "__main__":
    main()
