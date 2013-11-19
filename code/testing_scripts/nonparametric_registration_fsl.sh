#!/bin/bash

clear

MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/FSL/normalization

subject=1

for dir in ${data_directory}/*/ 
do
	
	rm anat_affine_fsl*

	if [ "$subject" -lt "199" ]
    then

		date1=$(date +"%s")

		# Parametric registration, default parameters
		flirt -in ${dir}/anat/mprage_skullstripped.nii.gz -ref ${MNI_TEMPLATE} -out anat_affine_fsl.nii

		# Non-parametric registration, default parameters
		fnirt --ref=${MNI_TEMPLATE} --in=anat_affine_fsl.nii.gz --iout=${results_directory}/FSL_warped_subject${subject}.nii

		date2=$(date +"%s")
		diff=$(($date2-$date1))
		echo "$(($diff))" >> fsl_normalization_times.txt

		subject=$((subject + 1))
	fi

	echo $subject

done

rm anat_affine_fsl*


