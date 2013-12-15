from nipype.interfaces.broccoli import motioncorrection
import os

BROCCOLI_DIR = '/home/miha/Programiranje/BROCCOLI'
OpenfMRI_DIR = '/data/miha/OpenfMRI/RhymeJudgment/ds003'
subject = 'sub00156'
SUBJECT_DIR = os.path.join(BROCCOLI_DIR, 'test_data/fcon1000/classic/Cambridge/', subject)

motioncorrection.MotionCorrection.help()
print(SUBJECT_DIR)

interface = motioncorrection.MotionCorrection(
    filters_parametric = BROCCOLI_DIR + '/code/Matlab_Wrapper/filters_for_parametric_registration.mat',   
    fMRI_file = os.path.join(SUBJECT_DIR, 'func/rest.nii.gz'),
    
    opencl_device = 0,
    show_results = True,
)


results = interface.run()
print(repr(results))

