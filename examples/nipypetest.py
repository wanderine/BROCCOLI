import nipype
import nipype.interfaces.broccoli as broccoli

from timeit import default_timer as timer

#info = broccoli.GetOpenCLInfo()
#info.run()

#bandwidth = broccoli.GetBandwidth()
#bandwidth.inputs.platform = 0
#bandwidth.inputs.device = 0
#bandwidth.run()

reg = broccoli.RegisterTwoVolumes()

reg.inputs.in_file = 'highres001_brain.nii'
reg.inputs.reference = 'MNI2mm.nii.gz'
reg.inputs.quiet = True
print reg.cmdline
start = timer()
reg.run()
end = timer()
print "Linear and non-linear registration took", end - start ,"seconds for 2 mm MNI template for OpenCL platform 0"

reg.inputs.in_file = 'highres001_brain.nii'
reg.inputs.reference = 'MNI1mm.nii.gz'
reg.inputs.quiet = True
print reg.cmdline
start = timer()
#reg.run()
end = timer()
print "Linear and non-linear registration took", end - start ,"seconds for 1 mm MNI template for OpenCL platform 0"


reg.inputs.platform = 1

reg.inputs.reference = 'MNI2mm.nii.gz'
print reg.cmdline
start = timer()
reg.run()
end = timer()
print "Linear and non-linear registration took", end - start ,"seconds for 2 mm MNI template for OpenCL platform 1"

reg.inputs.reference = 'MNI1mm.nii.gz'
print reg.cmdline
start = timer()
#reg.run()
end = timer()
print "Linear and non-linear registration took", end - start ,"seconds for 1 mm MNI template for OpenCL platform 1"

sm = broccoli.Smoothing()
sm.inputs.in_file = 'bold.nii'
sm.inputs.quiet = True
print sm.cmdline
start = timer()
sm.run()
end = timer()
print "Smoothing took", end - start ,"seconds"


sm.inputs.output = 'smoothed'
print sm.cmdline
start = timer()
sm.run()
end = timer()
print "Smoothing took", end - start ,"seconds"

sm.inputs.output = 'smoothed32.nii.gz'
print sm.cmdline
start = timer()
sm.run()
end = timer()
print "Smoothing took", end - start ,"seconds"


mc = broccoli.MotionCorrection()
mc.inputs.in_file = 'bold.nii'
mc.inputs.quiet = True
print mc.cmdline
start = timer()
mc.run()
end = timer()
print "Motion correection took", end - start ,"seconds"

mc.inputs.output = 'test2.nii'
print mc.cmdline
start = timer()
mc.run()
end = timer()
print "Motion correection took", end - start ,"seconds"

mc.inputs.output = 'test3.nii.gz'
print mc.cmdline
start = timer()
mc.run()
end = timer()
print "Motion correection took", end - start ,"seconds"

