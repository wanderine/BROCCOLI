#!/bin/bash

clear

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion
results_directory=/data/andek/BROCCOLI_test_data/FSL/motion_correction

date1=$(date +"%s")

for subject in {1..198}
do

	echo $subject

	#{ time mcflirt -in ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_no_noise.nii -refvol 0  -plots -out ${results_directory}/FSL_motion_corrected_subject${subject}_random_motion_no_noise.nii ; } 2>> fsl_motion_correction_times.txt

	#mcflirt -in ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_no_noise.nii -refvol 0  -plots -out ${results_directory}/FSL_motion_corrected_subject${subject}_random_motion_no_noise.nii

	#mcflirt -in ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_2percent_noise.nii -refvol 0  -plots -out ${results_directory}/FSL_motion_corrected_subject${subject}_random_motion_2percent_noise.nii

	mcflirt -in ${data_directory}/cambridge_rest_subject_${subject}_with_random_motion_shading.nii -refvol 0  -plots -out ${results_directory}/FSL_motion_corrected_subject${subject}_random_motion_shading.nii
	
done

date2=$(date +"%s")
diff=$(($date2-$date1)) 
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."


