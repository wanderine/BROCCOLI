from nipype.interfaces.broccoli import registration

BROCCOLI_DIR = '/home/miha/Programiranje/BROCCOLI'

registration.RegistrationT1MNI.help()

reg = registration.RegistrationT1MNI(
    mni_file = BROCCOLI_DIR + '/brain_templates/MNI152_T1_2mm.nii',
    t1_file = BROCCOLI_DIR + '/test_data/fcon1000/classic/Cambridge/sub00156/anat/mprage_skullstripped.nii.gz',
    filters_parametric = BROCCOLI_DIR + '/code/Matlab_Wrapper/filters_for_parametric_registration.mat',
    filters_nonparametric = BROCCOLI_DIR + '/code/Matlab_Wrapper/filters_for_nonparametric_registration.mat',
    
    show_results = True,
)

results = reg.run()
print(results)

reg = registration.RegistrationEPIT1(
    epi_file = BROCCOLI_DIR + '/test_data/fcon1000/classic/Cambridge/sub00156/func/rest.nii.gz',
    t1_file = BROCCOLI_DIR + '/test_data/fcon1000/classic/Cambridge/sub00156/anat/mprage_skullstripped.nii.gz',
    filters_parametric = BROCCOLI_DIR + '/code/Matlab_Wrapper/filters_for_parametric_registration.mat',
    filters_nonparametric = BROCCOLI_DIR + '/code/Matlab_Wrapper/filters_for_nonparametric_registration.mat',
    
    show_results = True,
)

results = reg.run()
print(results)

