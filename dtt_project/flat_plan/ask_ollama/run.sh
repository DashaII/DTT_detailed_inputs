#!/bin/bash
#SBATCH -J general_run		# name of job
#SBATCH -p gpu      		# name of partition or queue (if not specified default partition is used)
#SBATCH --gpus=1                # number of GPUs to request (default 0)
#SBATCH --mem=64G               # request 16 gigabytes memory (per node, default depends on node)
#SBATCH -o general_out.out    # name of output file for the job
#SBATCH -e general_err.err    # name of error file for the job

# here start the actual commands
ollama serve &  # Run Ollama in the background
sleep 5         # Give some time for the server to start
python3 script/ask_ollama/ask_ollama.py  # Run python script
