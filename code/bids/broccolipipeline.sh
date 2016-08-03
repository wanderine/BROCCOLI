#!/bin/bash

# check that we have at least 3 arguments

if [ $# -lt 3 ]; then
    echo "usage: broccolipipeline bids_dir output_dir analysis_type [participant(s)]"
    exit 1
fi

# first argument, bids_dir
bids_dir=$1

# second argument output_dir
output_dir=$2

# third argument, participant or group
analysis_type=$3

# check if analysis type is valid
if [ "$analysis_type" == "participant" ]; then
    echo "Analyzing single participant"
elif [ "$analysis_type" == "group" ]; then
    echo "Doing group analysis"
else
    echo "analysis_type must be 'participant' or 'group'"
    exit 1
fi

single_subject=0

# fourth optional argument, participant label
if [ $# -ge 4 ]; then
    fourth_argument=$4

    if [ "$fourth_argument" != "--participant-label" ]; then
        echo "Fourth argument must be '--participant-label'"
        exit 1
    else
        single_subject=1
    fi
fi


if [ $# -eq 4 ]; then
    echo "participant-label cannot be empty!"
    exit 1
fi

# participant label
if [ $# -eq 5 ]; then
    participant=$5
fi

if [ $# -eq 3 ]; then
    echo "bids_dir is $bids_dir, output_dir is $output_dir, analysis type is $analysis_type"
elif [ $# -eq 5 ]; then
    echo "bids_dir is $bids_dir, output_dir is $output_dir, analysis type is $analysis_type, participant is $participant"
fi

studyname=rhymejudgment

# Run the actual first level analysis

# participant given, analyze single subject
if [ "$single_subject" -eq "1" ]; then

    subject=sub-$participant

    mkdir $output_dir/$subject

    # convert BIDS events to FSL standard
    ./BIDSto3col.sh $bids_dir/$subject/func/${subject}_task-${studyname}_events.tsv $output_dir/$subject/cond

    # count number of trial types
    num_trial_types=`ls $output_dir/$subject/cond* | wc -l`
    cond_files=`ls $output_dir/$subject/cond*`

    Files=()
    string=${cond_files[$((0))]}
    Files+=($string)

    echo "Cond files are $cond_files"

    ((num_trial_types--))

    for f in $(seq 0 $num_trial_types); do
        # Get current file
        File=${Files[$((f))]}
        # Get number of events
        events=`cat $File | wc -l`
        ((events--))
        ((events++))
        echo "File is $File"
        echo "Number of events is $events"
        sed -i "1s/^/NumEvents $events \n\n/" $File
   done

    ((num_trial_types++))

    # create regressors file in BROCCOLI format
    touch $output_dir/$subject/regressors.txt
    echo "NumRegressors $num_trial_types" > $output_dir/$subject/regressors.txt
    echo "" >> $output_dir/$subject/regressors.txt

    # add cond files to regressors.txt
    ls $output_dir/$subject/cond* >> $output_dir/$subject/regressors.txt

    # create contrasts file in BROCCOLI format
    touch $output_dir/$subject/contrasts.txt
    echo "NumRegressors $num_trial_types" > $output_dir/$subject/contrasts.txt
    echo "NumContrasts $num_trial_types" >> $output_dir/$subject/contrasts.txt

    # make brain segmentation
    bet $bids_dir/$subject/anat/${subject}_T1w.nii.gz $output_dir/$subject/${subject}_T1w_brain.nii.gz

    #fslreorient2std

    FirstLevelAnalysis $bids_dir/$subject/func/${subject}_task-${studyname}_bold.nii.gz $output_dir/$subject/${subject}_T1w_brain.nii.gz MNI152_T1_2mm_brain.nii.gz $output_dir/$subject/regressors.txt $output_dir/$subject/contrasts.txt -output $output_dir/$subject -device 2

# participant not given, analyze all subjects
else
    # get number of subjects
    number_of_subjects=`cat $bids_dir/participants.tsv  | wc -l`
    ((number_ob_subjects--))

    for subject in $(seq 1 $number_of_subjects); do

        mkdir $output_dir/$subject

        # create regressors file in BROCCOLI format
        touch $output_dir/$subject/regressors.txt

        # create contrasts file in BROCCOLI format
        touch $output_dir/$subject/contrasts.txt

        echo "Subject $subject"
    done
fi





