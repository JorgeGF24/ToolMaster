#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jg2619@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/toolformer/cache
/usr/bin/nvidia-smi
echo $(date)
echo "Starting 48Gb job"
SECONDS=0
source activate
pip install transformers[deepspeed]
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


##########3 ssh cloud-vm-45-11.doc.ic.ac.uk