#!/bin/bash

clear

export OMP_NUM_THREADS=1
MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/AFNI/normalization

#interpolation=1 # Linear
interpolation=2 # sinc

subject=1

# Unifize intensity of MNI template
rm MNI_unifized.nii
3dUnifize -GM -prefix MNI_unifized.nii -input ${MNI_TEMPLATE}
UNIFIZED_MNI_TEMPLATE=MNI_unifized.nii

for dir in ${data_directory}/*/ 
do
	rm anat_unifized.nii
	rm anat_affine.nii
	
	if [ "$subject" -lt "199" ]
    then		

		date1=$(date +"%s")

		# The pipeline and parameters were obtained from the help text for 3dQwarp

		# Unifize intensity		
		3dUnifize -GM -prefix anat_unifized.nii -input ${dir}/anat/mprage_skullstripped.nii.gz		

		# Parametric registration, linear interpolation by default for optimization, cubic interpolation for final transformation
		3dAllineate -prefix anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE} -source anat_unifized.nii -source_automask -twopass -cost lpa -1Dmatrix_save ${results_directory}/anat_affine_subject${subject}.1D -autoweight -fineblur 3 -cmass 
		
		# Non-parametric registration, uses cubic and Hermite quintic basis functions
		3dQwarp -duplo -useweight -nodset -blur 0 3 -prefix ${results_directory}/AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE}    		

		# Linear
		if [ "$interpolation" -eq "1" ]
		then

			# Apply found transformations to original volume (without unifized intensity) using linear interpolation
			3dNwarpApply -interp linear -nwarp ${results_directory}/AFNI_warped_subject${subject}_WARP.nii -affter ${results_directory}/anat_affine_subject${subject}.1D -source ${dir}/anat/mprage_skullstripped.nii.gz -master ${MNI_TEMPLATE} -prefix ${results_directory}/AFNI_warped_subject${subject}.nii 

		elif [ "$interpolation" -eq "2" ]
		then

			# Apply found transformations to original volume (without unifized intensity) using sinc interpolation (default)
			3dNwarpApply -nwarp ${results_directory}/AFNI_warped_subject${subject}_WARP.nii -affter ${results_directory}/anat_affine_subject${subject}.1D -source ${dir}/anat/mprage_skullstripped.nii.gz -master ${MNI_TEMPLATE} -prefix ${results_directory}/AFNI_warped_subject${subject}_sinc.nii 	

		fi

		date2=$(date +"%s")
		diff=$(($date2-$date1))
		echo "$(($diff))" >> afni_normalization_times.txt

		subject=$((subject + 1))	
	fi

	rm anat_unifized.nii
	rm anat_affine.nii

done






