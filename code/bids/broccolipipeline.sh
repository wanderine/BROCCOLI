#!/bin/bash
set -e

function analyze_subject {

    bids_dir=$1
    output_dir=$2
    subject=$3
    task_name=$4

    one=1

    single_run=1
    if [ -e "${bids_dir}/${subject}/func/${subject}_task-${task_name}_run-01_events.tsv" ]; then
	single_run=0
    fi

    num_runs=1
    # Get number of runs
    if [ "${single_run}" -eq "0" ]; then
	num_runs=`ls ${bids_dir}/${subject}/func/${subject}_task-${task_name}_run-*_events.tsv | wc -l`
    fi
    echo -e "Number of runs is $num_runs \n"

    # convert BIDS csv to FSL format

    # Single run
    if [ "${single_run}" -eq "1" ]; then
        /Downloads/BROCCOLI/code/bids/BIDSto3col.sh ${bids_dir}/${subject}/func/${subject}_task-${task_name}_events.tsv ${output_dir}/${subject}/${task_name}/cond_run1
    # Several runs
    elif [ "${single_run}" -eq "0" ]; then
        for r in $(seq 1 $num_runs); do
            /Downloads/BROCCOLI/code/bids/BIDSto3col.sh ${bids_dir}/${subject}/func/${subject}_task-${task_name}_run-0${r}_events.tsv ${output_dir}/${subject}/${task_name}/cond_run${r}
        done
    else
	echo "Unknown number of runs!"
    fi

    # Remove any "na" files
    rm -f ${output_dir}/${subject}/${task_name}/cond_run*_na.txt

    # Check if all event types are present for all runs
    if [ "${single_run}" -eq "0" ]; then

	all_runs_equal=1
	max_run=1
        
	# Check number of event types for first run		
	num_trial_types_run1=`ls ${output_dir}/${subject}/${task_name}/cond_run1* | wc -l`
	max_num_trial_types=$num_trial_types_run1
	# Check number of event types for all other runs
        for r in $(seq 2 $num_runs); do        
            num_trial_types=`ls ${output_dir}/${subject}/${task_name}/cond_run${r}* | wc -l`
            if [ "$num_trial_types_run1" -ne "$num_trial_types" ]; then
		all_runs_equal=0
	    fi
	    # Save run with largest number of event types
	    if [ "$num_trial_types" -gt "$max_num_trial_types" ]; then
	       max_run=$r
	       max_num_trial_types=$num_trial_types
	    fi
	done

	# Add missing cond files
	if [ "${all_runs_equal}" -eq "0" ]; then

	    # Get all event names for run with largest number of event types
	    cond_files=`ls ${output_dir}/${subject}/${task_name}/cond_run${max_run}* | grep -oP "run${max_run}_([a-zA-Z0-9-_]+)" | sed -e "s/run${max_run}_/@/g" | cut -d "@" -f 2`
	    event_names=()
	    string=${cond_files[$((0))]}
	    event_names+=($string)
	
	    for r in $(seq 1 $num_runs); do        

	        # Skip run with largest number of event types
		if [ "$r" -ne "$max_run" ] ; then

		    # Compare with run with largest number of event types
		    ((max_num_trial_types--))
		    # Check if each event type exists
		    for t in $(seq 0 $max_num_trial_types); do        
		        event_name=${event_names[$((t))]}
			if [ ! -e ${output_dir}/${subject}/${task_name}/cond_run${r}_${event_name}.txt ] ; then
			    touch ${output_dir}/${subject}/${task_name}/cond_run${r}_${event_name}.txt
			    echo "Adding dummy file for run $r event ${event_name}"
			fi						
		    done
		    ((max_num_trial_types++))

		fi		
	    done
	fi
    fi

    # Modify the cond files to BROCCOLI format
    for r in $(seq 1 $num_runs); do        
        num_trial_types=`ls ${output_dir}/${subject}/${task_name}/cond_run${r}* | wc -l`
        cond_files=`ls ${output_dir}/${subject}/${task_name}/cond_run${r}*`
        Files=()
        string=${cond_files[$((0))]}
        Files+=($string)

	# Put NumEvents into each cond file
        ((num_trial_types--))        
        for f in $(seq 0 $num_trial_types); do
            # Get current file
            File=${Files[$((f))]}
            # Get number of events
            events=`cat $File | wc -l`
 	    # Check if file is a dummy file
	    if [ "$events" -eq "0" ] ; then
	        echo "NumEvents 0" >> $File
		echo "" >> $File
	    else
	        sed -i "1s/^/NumEvents $events \n\n/" $File
	    fi
        done
        ((num_trial_types++))

        # create regressors file in BROCCOLI format
        touch ${output_dir}/${subject}/${task_name}/regressors_run${r}.txt
        echo "NumRegressors $num_trial_types" > ${output_dir}/${subject}/${task_name}/regressors_run${r}.txt
        echo "" >> ${output_dir}/${subject}/${task_name}/regressors_run${r}.txt
        # add cond files to regressors.txt
        ls ${output_dir}/${subject}/${task_name}/cond_run${r}* >> ${output_dir}/${subject}/${task_name}/regressors_run${r}.txt
    done

    # create contrasts file in BROCCOLI format
    touch ${output_dir}/${subject}/${task_name}/contrasts.txt
    echo "NumRegressors $num_trial_types" > ${output_dir}/${subject}/${task_name}/contrasts.txt
    echo "NumContrasts 1" >> ${output_dir}/${subject}/${task_name}/contrasts.txt

    num_zeros=$((num_trial_types-one))
    zeros=""
    for f in $(seq 1 $num_zeros); do
        zeros="$zeros 0"
    done
    echo "1 $zeros" >> ${output_dir}/${subject}/${task_name}/contrasts.txt


    # Run the actual analysis

    # Single run
    if [ "${single_run}" -eq "1" ]; then
        FirstLevelAnalysis ${bids_dir}/${subject}/func/${subject}_task-${task_name}_bold.nii.gz ${output_dir}/${subject}/${subject}_T1w_brain.nii.gz /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz ${output_dir}/${subject}/${task_name}/regressors_run1.txt ${output_dir}/${subject}/${task_name}/contrasts.txt -output ${output_dir}/${subject}/${task_name}/${subject} -device 0 -savemnimask -saveallaligned -savedesignmatrix -saveoriginaldesignmatrix -regressmotion
    # Several runs
    elif [ "${single_run}" -eq "0" ]; then

        # Put all bold files and all regressor files into one string
        bold_files=""
        regressor_files=""
        for r in $(seq 1 $num_runs); do
            bold_files="$bold_files ${bids_dir}/${subject}/func/${subject}_task-${task_name}_run-0${r}_bold.nii.gz" 
            regressor_files="$regressor_files  ${output_dir}/${subject}/${task_name}/regressors_run${r}.txt"
        done

        FirstLevelAnalysis -runs ${num_runs} ${bold_files} ${output_dir}/${subject}/${subject}_T1w_brain.nii.gz /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz ${regressor_files} ${output_dir}/${subject}/${task_name}/contrasts.txt -output ${output_dir}/${subject}/${task_name}/${subject} -device 0 -savemnimask -saveallaligned -savedesignmatrix -saveoriginaldesignmatrix -regressmotion
    fi
}







# check that we have at least 3 arguments

if [ $# -lt 3 ]; then
    echo "usage: broccolipipeline bids_dir output_dir analysis_type [participant(s)]"
    exit 1
fi

zero=0
one=1
ten=10

# first argument, bids_dir
bids_dir=$1

# second argument output_dir
output_dir=$2

# third argument, participant or group
analysis_type=$3

# Run the validator (if it exists) for the BIDS directory
if [ -e "/usr/bin/bids-validator" ]; then
    echo "Running the BIDS validator for the dataset"
	/usr/bin/bids-validator ${bids_dir}
else
	echo "Could not find BIDS validator!"
fi


# check if analysis type is valid
if [ "${analysis_type}" == "participant" ]; then
    echo -e "\nDoing first level analysis \n"
elif [ "${analysis_type}" == "group" ]; then
    echo -e "\nDoing group analysis \n"
else
    echo "analysis_type must be 'participant' or 'group'"
    exit 1
fi

# fourth optional argument, participant label(s)
if [ $# -ge 4 ]; then
    fourth_argument=$4

    if [ "$fourth_argument" != "--participant_label" ]; then
        echo "Fourth argument must be '--participant_label'"
        exit 1
    fi
fi

if [ $# -eq 4 ]; then
    echo "participant_label cannot be empty!"
    exit 1
fi

# Analyze some subjects
if [ $# -ge 5 ]; then
    # Get participant label(s)
    temp="${@:5}"
    participants=()
    string=${temp[$((0))]}
    participants+=($string)
    num_subjects=`echo ${#participants[@]}`
# Analyze all subjects
else
    num_subjects=`cat ${bids_dir}/participants.tsv  | wc -l`
    ((num_subjects--))
    participants=()
    for s in $(seq 1 ${num_subjects}); do
	if [ "$s" -lt "10" ] ; then
  	    participants+=(0$s)
	else
	    participants+=($s)
	fi
    done
fi


echo -e "\nbids_dir is ${bids_dir}, output_dir is ${output_dir}, analysis type is ${analysis_type}, participants are ${participants[@]} \n"


# Get number of task names
num_tasks=`find ${bids_dir} -maxdepth 1 -name "*bold*" | grep -oP "task-([a-zA-Z0-9]+)" | cut -d "-" -f 2 | uniq | wc -l`
((num_tasks--))

# Get all task names
temp=`find ${bids_dir} -maxdepth 1 -name "*bold*" | grep -oP "task-([a-zA-Z0-9]+)" | cut -d "-" -f 2 | uniq`
task_names=()
string=${temp[$((0))]}
task_names+=($string)

echo -e "\nTask names are \n"
for t in $(seq 0 ${num_tasks}); do
    task_name=${task_names[$((t))]}
    echo -e "${task_name}"
done
echo -e "\n"

((num_subjects--))

# First level analysis
if [ "${analysis_type}" == "participant" ]; then

    for s in $(seq 0 ${num_subjects}); do

	subject=sub-${participants[$((s))]}

        echo -e "\n\nAnalyzing subject ${subject}\n\n"

  	# Make a new directory
        mkdir ${output_dir}/${subject}

   	# Check if anatomical volume exists
	if [ -e "${bids_dir}/${subject}/anat/${subject}_T1w.nii.gz" ]; then

            # make brain segmentation
            /usr/local/fsl/bin/bet ${bids_dir}/${subject}/anat/${subject}_T1w.nii.gz ${output_dir}/${subject}/${subject}_T1w_brain.nii.gz

   	    # Run analyze_subject once per task
    	    for t in $(seq 0 ${num_tasks}); do
 	        task_name=${task_names[$((t))]}
    	        echo -e "\n\nAnalyzing subject ${subject} task ${task_name} \n\n"
	        # Check if BOLD file exists
	        if [ ! -e "${bids_dir}/${subject}/func/${subject}_task-${task_name}_bold.nii.gz" ]  && [ ! -e "${bids_dir}/${subject}/func/${subject}_task-${task_name}_run-01_bold.nii.gz" ] ; then
	            echo "BOLD file does not exist for subject ${subject} task ${task_name}, skipping analysis"
	        else
                    mkdir ${output_dir}/${subject}/${task_name}
	            analyze_subject ${bids_dir} ${output_dir} ${subject} ${task_name}
	        fi
            done
        else
            echo "T1w file does not exist for subject ${subject}, skipping analysis"
	fi    	
    done
# Group analysis
elif [ "${analysis_type}" == "group" ]; then

    mkdir ${output_dir}/groupanalyses

    # Run group analysis for each task
    for t in $(seq 0 ${num_tasks}); do

	task_name=${task_names[$((t))]}
        echo -e "\n\nAnalyzing task ${task_name} \n\n"
        mkdir ${output_dir}/groupanalyses/${task_name}

	# Merge copes from first level analysis
	allcopes=""
	for s in $(seq 0 ${num_subjects}); do
	    subject=sub-${participants[$((s))]}
	    if [ -e "${output_dir}/${subject}/${task_name}/${subject}_cope_contrast0001_MNI.nii" ]; then
	        allcopes="$allcopes ${output_dir}/${subject}/${task_name}/${subject}_cope_contrast0001_MNI.nii"
	    else
	        echo "Could not find ${output_dir}/${subject}/${task_name}/${subject}_cope_contrast0001_MNI.nii, skipping subject"
	    fi
    	done

	/usr/local/fsl/bin/fslmerge -t ${output_dir}/groupanalyses/${task_name}/copes $allcopes

	# Run permutation test
	RandomiseGroupLevel ${output_dir}/groupanalyses/${task_name}/copes.nii.gz -groupmean -mask /usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz 
    done
fi





