Download the Rhyme Judgment dataset from OpenfMRI, https://openfmri.org/dataset/ds000003/

```
wget https://s3.amazonaws.com/openneuro/ds000003/ds000003_R2.0.2/compressed/ds000003_R2.0.2_raw.zip
```

Then unzip the file

```
unzip ds000003_R2.0.2_raw.zip
```

The data are stored in BIDS format, so convert the timing information to FSL/BROCCOLI format using

```
./BIDSto3col.sh ds000003_R2.0.2/sub-01/func/sub-01_task-rhymejudgment_events.tsv tasks
```

this will give you two files, tasks_pseudoword.txt and tasks_word.txt, which contain the start time of each event, the length of each event
and the value to use in the design matrix (normally 1). We now need to add the number of events to the top of these files for BROCCOLI. 

Open each file in a text editor and add "NumEvents 32" at the top, or in Linux do 

```
sed -i "1s/^/NumEvents 32 \n\n/" tasks_pseudoword.txt

sed -i "1s/^/NumEvents 32 \n\n/" tasks_word.txt
```

We now need to setup a text file that contains the name of these regressor files, and a file that contains the contrasts of interest. See regressors.txt and contrasts.txt.

To do a brain segmentation of the T1 volume we can for example run bet in FSL

```
bet ds000003_R2.0.2/sub-01/anat/sub-01_T1w.nii.gz ds000003_R2.0.2/sub-01/anat/sub-01_T1w_brain.nii.gz
```

A first level analysis can now be launched as

```
FirstLevelAnalysis ds000003_R2.0.2/sub-01/func/sub-01_task-rhymejudgment_bold.nii.gz  ds000003_R2.0.2/sub-01/anat/sub-01_T1w_brain.nii.gz /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz regressors.txt contrasts.txt -output sub-01
```

The resulting t-scores are stored as

sub-01_tscores_contrast*

and the resulting beta weights for the regressors in the design matrix are stored as

sub-01_beta_regressor*

