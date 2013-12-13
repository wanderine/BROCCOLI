#!/bin/bash

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# This script is only used to measure the processing time for normalization to a 1 mm template (there is no T1_2_MNI152_1mm.cnf)
#------------------------------------------------------------------------------------------------------------------------------------------------------------

clear

MNI_TEMPLATE_BRAIN=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_1mm.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge
results_directory=/data/andek/BROCCOLI_test_data/FSL/normalization_speed

subject=1

for dir in ${data_directory}/*/ 
do
	
	rm affine_parameters_fsl.mat

	if [ "$subject" -lt "199" ]
    then

		date1=$(date +"%s")

		# Parametric registration (skullstripped)
		flirt -omat affine_parameters_fsl.mat  -in ${dir}/anat/mprage_skullstripped.nii.gz -ref ${MNI_TEMPLATE_BRAIN}

		# Non-parametric registration (with skull)
		fnirt --ref=${MNI_TEMPLATE} --aff=affine_parameters_fsl.mat --in=${dir}/anat/mprage_anonymized.nii.gz --iout=${results_directory}/FSL_warped_subject${subject}_fnirt.nii	

		date2=$(date +"%s")
		diff=$(($date2-$date1))
		echo "$(($diff))" >> fsl_normalization_times.txt

		subject=$((subject + 1))

		echo $subject
	
		rm affine_parameters_fsl.mat

	fi


done



