
#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N VA_RNN              
#$ -cwd                  
#$ -l h_rt=01:00:00 
#$ -l h_vmem=16G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
source activate AMTFinalProj

# Run the program
python VA_RNN.py