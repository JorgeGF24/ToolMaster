#!/bin/bash
#SBATCH -o ./slurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=24mokies2@gmail.com # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export CUDA_HOME=/bin/cuda/12.2.0/
export HF_HOME=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
/usr/bin/nvidia-smi
echo $(date)
echo "Starting 48Gb job"
SECONDS=0
source activate
deepspeed --num_gpus=1 test_trained_model.py --deepspeed /vol/bitbucket/jg2619/augmenting_llms/model_training/model_experiments/ds_conf.json
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


##########3 ssh cloud-vm-45-11.doc.ic.ac.uk