#!/bin/bash
#SBATCH --gres=gpu:1
##SBATCH --partition=gpgpuB
#SBATCH -o ./zslurm/slurm-%j.out # STDOUT
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=24mokies2@gmail.com # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
echo $(date)
SECONDS=0
source activate
export PYTHONPATH=/vol/bitbucket/jg2619/augmenting_llms/:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
echo $PYTHONPATH
# Datasets separated by ", ". 
# Datasets: "test, asdiv, gms8k-easy, gms8k-hard, triviaQA"
# Models: "AY", "DX", "DX-2", "A basic 0-shot", "A basic 0-shot-b", "A basic 1-shot"" ", DX, DX-2, A basic 0-shot, A basic 0-shot-b, A basic 1-shot
python eval_benchmark.py "asdiv, triviaQA-small" "DZ-5.2-COT" "$1" # "test, asdiv, gms8k-easy, triviaQA" "DX, DX-2" "$1"
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."