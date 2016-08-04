#!/bin/bash

function analyze_subject {

    bids_dir=$1
    output_dir=$2
    studyname=$3
    subject=$4

    one=1

    mkdir $output_dir/$subject

    # convert BIDS csv to FSL format
    ./BIDSto3col.sh $bids_dir/$subject/func/${subject}_task-${studyname}_events.tsv $output_dir/$subject/cond

    # count number of trial types
    num_trial_types=`ls $output_dir/$subject/cond* | wc -l`
    cond_files=`ls $output_dir/$subject/cond*`

    Files=()
    string=${cond_files[$((0))]}
    Files+=($string)

    ((num_trial_types--))

    for f in $(seq 0 $num_trial_types); do
        # Get current file
        File=${Files[$((f))]}
        # Get number of events
        events=`cat $File | wc -l`
        ((events--))
        ((events++))
        #echo "File is $File"
        #echo "Number of events is $events"
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
    echo "NumContrasts $((num_trial_types-one))" >> $output_dir/$subject/contrasts.txt

    num_zeros=$((num_trial_types-one))
    zeros=""
    for f in $(seq 1 $num_zeros); do
        zeros="$zeros 0"
    done
    echo "1 $zeros" >> $output_dir/$subject/contrasts.txt

    # make brain segmentation
    /usr/local/fsl/bin/bet $bids_dir/$subject/anat/${subject}_T1w.nii.gz $output_dir/$subject/${subject}_T1w_brain.nii.gz

    #fslreorient2std $output_dir/$subject/${subject}_T1w_brain.nii.gz

    FirstLevelAnalysis $bids_dir/$subject/func/${subject}_task-${studyname}_bold.nii.gz $output_dir/$subject/${subject}_T1w_brain.nii.gz /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz $output_dir/$subject/regressors.txt $output_dir/$subject/contrasts.txt -output $output_dir/$subject/$subject -device 0 -savemnimask -saveallaligned

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

if [ "$analysis_type" == "participant" ]; then
    # participant given, analyze single subject
    if [ "$single_subject" -eq "1" ]; then
        subject=sub-$participant
        echo -e "\n\nAnalyzing subject $subject\n\n"
        analyze_subject $bids_dir $output_dir $studyname $subject
    # participant not given, analyze all subjects
    else
        # get number of subjects
        num_subjects=`cat $bids_dir/participants.tsv  | wc -l`
        ((num_subjects--))

        for s in $(seq 1 $num_subjects); do
            if [ "$s" -lt "$ten" ]; then
                subject=sub-$zero$s
            else
                subject=sub-$s
            fi
            echo -e "\n\nAnalyzing subject $subject\n\n"
            analyze_subject $bids_dir $output_dir $studyname $subject
        done
    fi
elif [ "$analysis_type" == "group" ]; then

    mkdir $output_dir/group

    # merge masks from first level analyses

    # get number of subjects
    num_subjects=`cat $bids_dir/participants.tsv  | wc -l`
    ((num_subjects--))

    allmasks=""
    for s in $(seq 1 $num_subjects); do
        if [ "$s" -lt "$ten" ]; then
            subject=sub-$zero$s
        else
            subject=sub-$s
        fi
        allmasks="$allmasks $output_dir/$subject/$subject_mask_mni.nii"
    done

    fslmerge -t $output_dir/group/mask $allmasks

    # merge copes from first level analyses

    allcopes=""
    for s in $(seq 1 $num_subjects); do
    if [ "$s" -lt "$ten" ]; then
        subject=sub-$zero$s
    else
        subject=sub-$s
    fi
        allcopes="$allcopes $output_dir/$subject/$subject_cope_contrast0001_MNI.nii"
    done

    fslmerge -t $output_dir/group/copes $allcopes


    RandomiseGroupLevel allcopes -groupmean -mask MNI152_T1_2mm_brain_mask.nii.gz -output $output_dir/group/

fi





