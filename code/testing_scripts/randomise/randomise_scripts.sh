time randomise -i betas49.nii.gz -d design1.mat -t design1.con -o permtest -n 9900 -m /home/andek/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz -N -x -P

#export FSLPARALLEL=condor
#time randomise_parallel -i betas49.nii.gz -d design1.mat -t design1.con -o permtest -n 9900 -m /home/andek/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz -N -x -P




