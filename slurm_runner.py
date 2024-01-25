import argparse
import subprocess
import os

def create_bash_script(script_name, arguments):
    with open(script_name, 'w') as script_file:
        script_file.write('#!/bin/bash\n')
        # Add any Slurm SBATCH directives here
        script_file.write(f'#SBATCH --partition={arguments.partition}\n')
        script_file.write(f'#SBATCH --account=LADE\n')
        script_file.write(f'#SBATCH --nodes={arguments.nodes}\n')
        script_file.write(f'#SBATCH --exclusive\n')
        script_file.write(f'#SBATCH --ntasks-per-node={arguments.ntasks_per_node}\n')
        script_file.write(f'#SBATCH --cpus-per-task={arguments.cpus_per_task}\n')
        script_file.write(f'#SBATCH --mem={arguments.mem}\n')
        script_file.write(f'#SBATCH --time={arguments.time}\n')
        script_file.write(f'#SBATCH --job-name={arguments.job_name}\n')
        script_file.write(f'#SBATCH --output=output_job/{arguments.output}_job_%j.out\n')
        # Add more SBATCH directives as needed

        # Example command to run
        script_file.write(f'echo "Running job: {arguments.job_name}"\n')
        # Add more commands based on your requirements



def main():
    parser = argparse.ArgumentParser(description="Create and submit a job to Slurm.")
    parser.add_argument('--job-name', type=str, help='Job name', required=True)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    # Add more arguments as needed

    args = parser.parse_args()

    script_name = 'generated_script.sh'
    create_bash_script(script_name, args)

    # Submit the script to Slurm
    subprocess.run(['sbatch', script_name])

if __name__ == "__main__":
    main()
