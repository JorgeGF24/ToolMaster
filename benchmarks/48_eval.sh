#!/bin/bash
#SBATCH -o ./zslurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=24mokies2@gmail.com # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export TORCH_USE_CUDA_DSA=1
/usr/bin/nvidia-smi
echo $(date)
SECONDS=0
source activate
export PYTHONPATH=/vol/bitbucket/jg2619/augmenting_llms/:$PYTHONPATH
echo "Evaluating Benchmarks on 48Gb job"
# Datasets separated by ", ". 
# Datasets: "test, asdiv, gms8k-easy, gms8k-hard, triviaQA"
# Models: "AY", "DX", "DX-2", "A basic 0-shot", "A basic 0-shot-b", "A basic 1-shot"
python eval_benchmark.py "ASDiv-full, triviaQA" "mickey-task, mickey-task-mono, just-gen, just-gen-mono" "$1" 
#  GPTJ_Master
#  "small-task, small-task-no-calc, small-task-mono, small-task-no-calc-mono" 
# med-arg, large, large-mono, large-low-k, large-high-k
# "med, med-mono, med-low-k" "GPTJ_baseline, GPTJ_Master" "$1"  
# "test" "small-task" "$1" #                   ##"AY, DX, DX-2, A basic 0-shot, A basic 0-shot-b, A basic 1-shot" "$1" # "asdiv-full, triviaQA" "GPTJ_baseline" # "asdiv, triviaQA-small" "mickey-task"
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


##########3 ssh cloud-vm-45-11.doc.ic.ac.uk

# 2248 is same as before but top k = 2