#!/bin/bash

STUDY=Cambridge
SUBJECT=sub00156

T1_FILE="../../test_data/fcon1000/classic/${STUDY}/${SUBJECT}/anat/mprage_skullstripped.nii"
MNI_FILE="../../brain_templates/MNI152_T1_2mm.nii"
EPI_FILE="../../test_data/fcon1000/classic/${STUDY}/${SUBJECT}/func/rest.nii"

PARAMETRIC_FILTERS_FILE="../Matlab_Wrapper/filters_for_parametric_registration.mat"
NONPARAMETRIC_FILTERS_FILE="../Matlab_Wrapper/filters_for_nonparametric_registration.mat"
FILTERS_ARGS="--filters-parametric ${PARAMETRIC_FILTERS_FILE} --filters-nonparametric ${NONPARAMETRIC_FILTERS_FILE}"

PYTHON=`which python2`
${PYTHON} RegisterT1MNI.py --t1-file ${T1_FILE} --mni-file ${MNI_FILE} ${FILTERS_ARGS}
${PYTHON} RegisterEPIT1.py --epi-file ${EPI_FILE} --t1-file ${T1_FILE} ${FILTERS_ARGS}
