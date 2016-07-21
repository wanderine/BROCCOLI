#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

# Set OpenCL package to use
AMD=0
INTEL=1
NVIDIA=2
#OPENCL_PACKAGE=$AMD
OPENCL_PACKAGE=$INTEL
#OPENCL_PACKAGE=$NVIDIA

# Set compilation mode to use
RELEASE=0
DEBUG=1
COMPILATION=$RELEASE

# Fist run make for Nifti library
#cd nifticlib-2.0.0
#make
#cd ..

# Need to install AMD OpenCL SDK first
if [ "$OPENCL_PACKAGE" -eq "$AMD" ]; then
    OPENCL_HEADER_DIRECTORY1=/opt/AMDAPP/include 
    OPENCL_HEADER_DIRECTORY2=/opt/AMDAPP/include/CL 
    OPENCL_LIBRARY_DIRECTORY=/opt/AMDAPP/lib/x86_64 
# Need to install Intel OpenCL SDK and Intel OpenCL runtime first
elif [ "$OPENCL_PACKAGE" -eq "$INTEL" ]; then
    OPENCL_HEADER_DIRECTORY1=/opt/intel/opencl-sdk/include 
    OPENCL_HEADER_DIRECTORY2=/opt/intel/opencl-sdk/include/CL
    OPENCL_LIBRARY_DIRECTORY=/opt/intel/opencl/lib64
# Need to install Nvidia CUDA SDK first
elif [ "$OPENCL_PACKAGE" -eq "$NVIDIA" ]; then
    OPENCL_HEADER_DIRECTORY1=/usr/local/cuda-6.5/include/CL
    OPENCL_HEADER_DIRECTORY2=
    #OPENCL_LIBRARY_DIRECTORY=/usr/lib64
	OPENCL_LIBRARY_DIRECTORY=/usr/local/cuda-6.5/lib64
else
    echo "Unknown OpenCL package!"
fi

CLBLAS_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/clBLASLinux

# Set compilation flags
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG -m64 -fopenmp"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Linux/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g -m64"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/compiled/BROCCOLI_LIB/Linux/Debug
else
    echo "Unknown compilation mode"
fi


g++ GetOpenCLInfo.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -lBROCCOLI_LIB -lOpenCL -lclBLAS ${FLAGS} -o GetOpenCLInfo &

g++ GetBandwidth.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -lBROCCOLI_LIB -lOpenCL -lclBLAS ${FLAGS} -o GetBandwidth &

# Support for compressed files
g++ MotionCorrection.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o MotionCorrection &

g++ RegisterTwoVolumes.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o RegisterTwoVolumes &

g++ TransformVolume.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o TransformVolume &

g++ RandomiseGroupLevel.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o RandomiseGroupLevel &

g++ FirstLevelAnalysis.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o FirstLevelAnalysis &

g++ SliceTimingCorrection.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o SliceTimingCorrection &

g++ Smoothing.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o Smoothing &

g++ GLM.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o GLM &

g++ ICA.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o ICA &

g++ Searchlight.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -L${OPENCL_LIBRARY_DIRECTORY} -L${CLBLAS_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz ${FLAGS} -o Searchlight &



#g++ CombineAffineTransforms.cpp -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -o CombineAffineTransforms &


#g++ MakeROI.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib  -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lniftiio -lznz -lz ${FLAGS} -o MakeROI &

#g++ ExtractTimeseries.cpp -I${OPENCL_HEADER_DIRECTORY1} -I${OPENCL_HEADER_DIRECTORY2} -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib  -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -lniftiio -lznz -lz ${FLAGS} -o ExtractTimeseries &

wait


# Move compiled files to correct directory
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
	mv GetOpenCLInfo ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv GetBandwidth ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv MotionCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv RegisterTwoVolumes ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv TransformVolume ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv RandomiseGroupLevel ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv FirstLevelAnalysis ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv SliceTimingCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv Smoothing ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv GLM ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv ICA ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	mv Searchlight ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	#mv MakeROI ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	#mv ExtractTimeseries ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
	#mv CombineAffineTransforms ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
	mv GetOpenCLInfo ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv GetBandwidth ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv MotionCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv RegisterTwoVolumes ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv TransformVolume ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv RandomiseGroupLevel ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv FirstLevelAnalysis ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv SliceTimingCorrection ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv Smoothing ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv GLM ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv ICA ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	mv Searchlight ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	#mv MakeROI ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	#mv ExtractTimeseries ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
	#mv CombineAffineTransforms ${BROCCOLI_GIT_DIRECTORY}/compiled/Bash/Linux/Debug
fi



# No support for compressed files
#g++ RegisterTwoVolumes.cpp ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c -lOpenCL -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o RegisterTwoVolumes

# Debugging
# gdb --args Program
# run

# Example debugging
# gdb --args ./FirstLevelAnalysis fMRI.nii T1_brain.nii MNI152_T1_1mm_brain.nii.gz regressors.txt  contrasts.txt -platform 2 -saveallaligned
# 'run' + enter
# 
# If segmentation fault, use 'backtrace' + enter (or 'backtrace full' to see values of local variables)
# Show a specific frame with 'frame number'
# exit with 'quit'




