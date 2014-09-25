#!/bin/bash

# Set compilation mode to use
RELEASE=0
DEBUG=1
COMPILATION=$RELEASE

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers

# Fist run make for Nifti library
#cd nifticlib-2.0.0
#make
#cd ..

# Set compilation flags
if [ "$COMPILATION" -eq "$RELEASE" ] ; then
    FLAGS="-O3 -DNDEBUG"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Release
elif [ "$COMPILATION" -eq "$DEBUG" ] ; then
    FLAGS="-O0 -g"
	BROCCOLI_LIBRARY_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug
else
    echo "Unknown compilation mode"
fi

# Compile each wrapper
g++ -framework OpenCL  GetOpenCLInfo.cpp -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen ${FLAGS} -o GetOpenCLInfo

g++ -framework OpenCL MotionCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o MotionCorrection

g++ -framework OpenCL RegisterTwoVolumes.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o RegisterTwoVolumes

g++ -framework OpenCL TransformVolume.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o TransformVolume

g++ -framework OpenCL RandomiseGroupLevel.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o TransformVolume

g++ -framework OpenCL FirstLevelAnalysis.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o FirstLevelAnalysis

g++ -framework OpenCL SliceTimingCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_LIBRARY_DIRECTORY} -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib ${FLAGS} -o SliceTimingCorrection

# Now compile the OpenCL kernel code using openclc
#/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -emit-llvm-bc broccoli_lib_kernel.cl -o broccoli_lib_kernel.bin


