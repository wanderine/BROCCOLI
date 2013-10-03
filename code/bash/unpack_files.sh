#!/bin/bash

clear


data_directory=/data/andek/BROCCOLI_test_data/Cambridge/

subject=1

for dir in ${data_directory}/*/ 
do
	if [ "$subject" -lt "199" ]
    then
	
		gzip -d ${dir}/anat/mprage_skullstripped.nii.gz
	
	fi

	subject=$((subject + 1))	

done

