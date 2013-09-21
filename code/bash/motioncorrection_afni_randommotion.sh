#!/bin/bash

clear

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion
results_directory=/data/andek/BROCCOLI_test_data/AFNI

subject=1

date1=$(date +"%s")

for file in ${data_directory}/*.nii
do

	echo $subject

	if [ "$subject" -lt "199" ]
    then

		3dvolreg -float -linear -prefix ${results_directory}/AFNI_motion_corrected_subject${subject}_random_motion.nii ${file}
	
	fi

	subject=$((subject + 1))	

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."


