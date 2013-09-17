#!/bin/bash

clear


MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/AFNI

subject=1

date1=$(date +"%s")

for dir in ${data_directory}/*/ 
do
	rm anat_unifized.nii
	rm anat_affine.nii
	rm anat_affine.1D
	rm AFNI_nonlinear.nii

	if [ "$subject" -eq "20" ]
    then

		# The pipeline and parameters were obtained from the help text for 3dQwarp

		# Unifize intensity
		3dUnifize -prefix anat_unifized.nii -input ${dir}/anat/mprage_skullstripped.nii.gz

		# Parametric registration
		3dAllineate -prefix anat_affine.nii -base ${MNI_TEMPLATE} -source anat_unifized.nii -twopass -cost lpa -1Dmatrix_save ${results_directory}/anat_affine_subject${subject}.1D -autoweight -fineblur 3 -cmass

		# Non-parametric registration
		#3dQwarp -duplo -useweight -prefix ${results_directory}/AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${MNI_TEMPLATE}
		3dQwarp -duplo -useweight -blur 0 3 -prefix ${results_directory}/AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${MNI_TEMPLATE}  

		# Remove resulting warped image
		rm ${results_directory}/AFNI_warped_subject${subject}.nii

		# Apply found transformations to original volume (without unifized intensity)
		3dNwarpApply -nwarp ${results_directory}/AFNI_warped_subject${subject}_WARP.nii -affter ${results_directory}/anat_affine_subject${subject}.1D -source ${dir}/anat/mprage_skullstripped.nii.gz -master ${MNI_TEMPLATE} -prefix ${results_directory}/AFNI_warped_subject${subject}.nii
	
	fi

	subject=$((subject + 1))	

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."


