#!/bin/bash

clear


MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/FSL

subject=1

date1=$(date +"%s")

for dir in ${data_directory}/*/
do

	echo $subject

	if [ "$subject" -lt "199" ]
    then

		mcflirt -in ${dir}/func/rest.nii.gz -refvol 0  -out ${results_directory}/FSL_motion_corrected_subject${subject}_original_motion.nii
	
	fi

	subject=$((subject + 1))	

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."


