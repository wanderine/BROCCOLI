#!/bin/bash

clear

export OMP_NUM_THREADS=1
MNI_TEMPLATE=/home/andek/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz

data_directory=/data/andek/BROCCOLI_test_data/Cambridge/
results_directory=/data/andek/BROCCOLI_test_data/AFNI

subject=1

date1=$(date +"%s")

# Unifize intensity of MNI template
rm MNI_unifized.nii
3dUnifize -GM -prefix MNI_unifized.nii -input ${MNI_TEMPLATE}
UNIFIZED_MNI_TEMPLATE=MNI_unifized.nii

for dir in ${data_directory}/*/ 
do
	rm anat_unifized.nii
	rm anat_affine.nii

	if [ "$subject" -lt "11" ]
    then

		# The pipeline and parameters were obtained from the help text for 3dQwarp

		# Unifize intensity		
		3dUnifize -GM -prefix anat_unifized.nii -input ${dir}/anat/mprage_skullstripped.nii		

		# Parametric registration
		#3dAllineate -prefix anat_affine.nii -base ${MNI_TEMPLATE} -source anat_unifized.nii -twopass -cost lpa -1Dmatrix_save ${results_directory}/anat_affine_subject${subject}.1D -autoweight -fineblur 3
		time 3dAllineate -prefix anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE} -source anat_unifized.nii -twopass -cost lpa -1Dmatrix_save ${results_directory}/anat_affine_subject${subject}.1D -autoweight -fineblur 3 -cmass
		#3dAllineate -prefix anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE} -source anat_unifized.nii -twopass -cost lpa -1Dmatrix_save anat_affine_subject${subject}.1D -autoweight -fineblur 3 -cmass

		# Non-parametric registration
		#3dQwarp -duplo -nodset -useweight -blur 0 3 -prefix ${results_directory}/AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${MNI_TEMPLATE}
		
		time 3dQwarp -duplo -useweight -nodset -blur 0 3 -prefix ${results_directory}/AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE}    
		#3dQwarp -duplo -useweight -nodset -blur 0 3 -prefix AFNI_warped_subject${subject}.nii -source anat_affine.nii -base ${UNIFIZED_MNI_TEMPLATE}    

		# Apply found transformations to original volume (without unifized intensity)
		#3dNwarpApply -nwarp ${results_directory}/AFNI_warped_subject${subject}_WARP.nii -affter ${results_directory}/anat_affine_subject${subject}.1D -source ${dir}/anat/mprage_skullstripped.nii.gz -master ${MNI_TEMPLATE} -prefix ${results_directory}/AFNI_warped_subject${subject}.nii

		time 3dNwarpApply -interp linear -nwarp ${results_directory}/AFNI_warped_subject${subject}_WARP.nii -affter ${results_directory}/anat_affine_subject${subject}.1D -source ${dir}/anat/mprage_skullstripped.nii -master ${MNI_TEMPLATE} -prefix ${results_directory}/AFNI_warped_subject${subject}_temp.nii

		#3dNwarpApply -interp linear -nwarp AFNI_warped_subject${subject}_WARP.nii -affter anat_affine_subject${subject}.1D -source mprage_skullstripped.nii.gz -master ${MNI_TEMPLATE} -prefix AFNI_warped_subject${subject}.nii
	
	fi

	subject=$((subject + 1))	

done

date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."





