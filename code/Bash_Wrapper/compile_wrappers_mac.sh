#!/bin/bash

# Set compilation mode to use
RELEASE=0
DEBUG=1
COMPILATION=$RELEASE

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
#OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers
OPENCL_HEADER_DIRECTORY=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/

# Fist run make for Nifti library
#cd nifticlib-2.0.0
#make
#cd ..

# Set compilation flags
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Mac/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Mac/Debug
else
    echo "Unknown compilation mode"
fi

# Compile each wrapper
g++ -framework OpenCL  GetOpenCLInfo.cpp -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -o GetOpenCLInfo

g++ -framework OpenCL  GetBandwidth.cpp -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -o GetBandwidth

g++ -framework OpenCL MotionCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o MotionCorrection

g++ -framework OpenCL RegisterTwoVolumes.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o RegisterTwoVolumes

g++ -framework OpenCL TransformVolume.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o TransformVolume

g++ -framework OpenCL RandomiseGroupLevel.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o RandomiseGroupLevel

g++ -framework OpenCL FirstLevelAnalysis.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o FirstLevelAnalysis -Wall

g++ -framework OpenCL SliceTimingCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o SliceTimingCorrection

g++ -framework OpenCL Smoothing.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o Smoothing

g++ -framework OpenCL GLM.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o GLM

g++ -framework OpenCL ICA.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o ICA

g++ -framework OpenCL Searchlight.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o Searchlight




# Move compiled files to correct directory

if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    mv GetOpenCLInfo ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv GetBandwidth ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv MotionCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv RegisterTwoVolumes ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv TransformVolume ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv RandomiseGroupLevel ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv FirstLevelAnalysis ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv SliceTimingCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv Smoothing ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv GLM ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv ICA ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
    mv Searchlight ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    mv GetOpenCLInfo ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv GetBandwidth ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv MotionCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv RegisterTwoVolumes ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv TransformVolume ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv RandomiseGroupLevel ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv FirstLevelAnalysis ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv SliceTimingCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv Smoothing ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv GLM ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv ICA ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
    mv Searchlight ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Mac/Debug
fi

# For debugging, use lldb

# Example
# lldb RegisterTwoVolumes
# r mprage_skullstripped.nii.gz MNI152_T1_2mm_brain.nii.gz -device 2, 'enter'

