#!/bin/bash

clear


MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/FSL

subject=1

date1=$(date +"%s")

for dir in ${data_directory}/*/ 
do
	#dir=/data/andek/BROCCOLI_test_data/Cambridge/sub04491
	
	rm anat_affine_fsl*

	flirt -in ${dir}/anat/mprage_skullstripped.nii.gz -ref ${MNI_TEMPLATE} -out anat_affine_fsl.nii
	fnirt --ref=${MNI_TEMPLATE} --in=anat_affine_fsl.nii.gz --iout=${results_directory}/FSL_warped_${subject}.nii
	
	subject=$((subject + 1))

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."

