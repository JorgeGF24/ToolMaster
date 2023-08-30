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
echo $2
python eval_benchmark.py $1 $2 $3
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


##########3 ssh cloud-vm-45-11.doc.ic.ac.uk