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
# TODO: med-no-token-no-tools-1, med-no-token-no-tools-2, med-no-token-no-calc-mono-high-k, med-no-token-mono-low-k, 
# Datasets: "test, asdiv, gms8k-easy, gms8k-hard, triviaQA"
# Models: "AY", "DX", "DX-2", "A basic 0-shot", "A basic 0-shot-b", "A basic 1-shot"
#python eval_benchmark.py "ASDiv-full, triviaQA" "med, med-no-token, med-no-token-high-k, med-no-token-low-k, med-no-token-mono, med-no-token-0" "$1"
python eval_perplexity.py 
#python eval_benchmark.py "ASDiv-full" "med-arg-0" "$1"
# "med-no-token, med-no-token-high-k, med-no-token-low-k, med-no-token-mono, med-no-token-0" "$1" 
# TODO: med-no-token-no-tools-1, med-no-token-no-tools-2
# TODO: GPTJ_Master-2shot on triviaqa in top 5 and 10 mode
#  GPTJ_Master
# TODO: GPTJ_Master-2shot, GPTJ_baseline+, 
# just-gen, just-gen-mono, just-gen-task, just-gen-mono-task, mickey, mickey-task, mickey-task-mono, small-task, small2, small2-high
# med, med-no-task, med-mono, med6, med-low-k, med-arg, med-arg6, med-arg6-high-k, med-no-calc, med-no-calc-mono, med-shuffle, large, huge 
# med, med6, med-mono, med-no-task, 
# med-arg6, med-arg6-low-k, med-arg6-high-k",
#  "med-full, med-no-calc, med-no-calc-mono" 
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