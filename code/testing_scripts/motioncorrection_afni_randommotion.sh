#!/bin/bash

clear

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion
results_directory=/data/andek/BROCCOLI_test_data/AFNI/motion_correction

date1=$(date +"%s")

for subject in {1..198}	
do

	echo $subject

	{ time 3dvolreg -float -linear -1Dfile ${results_directory}/AFNI_motion_parameters_subject${subject}_random_motion_no_noise.1D -prefix ${results_directory}/AFNI_motion_corrected_subject${subject}_random_motion_no_noise.nii ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_no_noise.nii ;} 2>> afni_motion_correction_times.txt

	#3dvolreg -float -linear -1Dfile ${results_directory}/AFNI_motion_parameters_subject${subject}_random_motion_no_noise.1D -prefix ${results_directory}/AFNI_motion_corrected_subject${subject}_random_motion_no_noise.nii ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_no_noise.nii

	#3dvolreg -float -linear -1Dfile ${results_directory}/AFNI_motion_parameters_subject${subject}_random_motion_1percent_noise.1D -prefix ${results_directory}/AFNI_motion_corrected_subject${subject}_random_motion_1percent_noise.nii ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_1percent_noise.nii

	#3dvolreg -float -linear -1Dfile ${results_directory}/AFNI_motion_parameters_subject${subject}_random_motion_2percent_noise.1D -prefix ${results_directory}/AFNI_motion_corrected_subject${subject}_random_motion_2percent_noise.nii ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_2percent_noise.nii

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
