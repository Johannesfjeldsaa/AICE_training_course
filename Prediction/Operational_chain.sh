#!/bin/bash -f
#$ -N AICE_Operational_chain
#$ -l h_rt=00:20:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=4G,mem_free=4G,h_data=10G
#$ -q research-r8.q
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y
 
module use /modules/MET/rhel8/user-modules/
module load cuda/11.6.0
 
source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate /lustre/storeB/users/cyrilp/mycondaTF

current_path="/lustre/storeB/users/cyrilp/ML_course_June_2024/Prediction/"
cd $current_path

python3 $current_path"AICE_predictors.py"
python3 $current_path"AICE_forecasts.py"
python3 $current_path"AICE_hdf5_to_netCDF.py"
