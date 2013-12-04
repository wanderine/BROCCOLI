#!/bin/bash

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# This script is used for normalization to a 2 mm template (there is no T1_2_MNI152_1mm.cnf), the estimated displacement field is upscaled to obtain
# a normalized 1 mm T1 volume
#------------------------------------------------------------------------------------------------------------------------------------------------------------

clear

MNI_TEMPLATE_BRAIN=/home/andek/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
MNI_TEMPLATE_BRAIN_1mm=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_2mm.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge
results_directory=/data/andek/BROCCOLI_test_data/FSL/normalization

subject=1

for dir in ${data_directory}/*/ 
do
	
	if [ "$subject" -lt "199" ]
    then

		date1=$(date +"%s")

		# Parametric registration (skullstripped)
		flirt -omat ${results_directory}/affine_parameters_subject${subject}.mat  -in ${dir}/anat/mprage_skullstripped.nii.gz -ref ${MNI_TEMPLATE_BRAIN} -out ${results_directory}/FSL_affine_subject${subject}.nii
 
		# Non-parametric registration (with skull)
		fnirt --aff=${results_directory}/affine_parameters_subject${subject}.mat --in=${dir}/anat/mprage_anonymized.nii.gz --config=/home/andek/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf --fout=${results_directory}/FSL_warping_field_subject${subject}.nii --iout=${results_directory}/FSL_warped_subject${subject}_fnirt.nii

		# Apply to skullstripped, 1 mm
		applywarp -i ${dir}/anat/mprage_skullstripped.nii.gz -o ${results_directory}/FSL_warped_subject${subject}.nii -r ${MNI_TEMPLATE_BRAIN_1mm} -w ${results_directory}/FSL_warping_field_subject${subject}.nii -s

		date2=$(date +"%s")
		diff=$(($date2-$date1))
		echo "$(($diff))" >> fsl_normalization_times.txt

		subject=$((subject + 1))
	fi

	echo $subject

done



