#!/bin/bash
#SBATCH -o ./slurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jg2619@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export HF_HOME=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export TORCH_EXTENSIONS_DIR=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export TORCH_CUDA_ARCH_LIST="8.6"
/usr/bin/nvidia-smi
echo $(date)
echo "Starting 48Gb job"
SECONDS=0
source activate
accelerate launch --config_file /vol/bitbucket/jg2619/augmenting_llms/model_training/model_experiments/a_conf.json train.py
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."



##########3 ssh cloud-vm-45-11.doc.ic.ac.uk