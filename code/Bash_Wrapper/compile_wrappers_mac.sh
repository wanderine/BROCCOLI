#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers

# Fist run make for Nifti library
cd nifticlib-2.0.0
make
cd ..

g++ -framework OpenCL  GetOpenCLInfo.cpp -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -o GetOpenCLInfo

g++ -framework OpenCL MotionCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o MotionCorrection

g++ -framework OpenCL RegisterTwoVolumes.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o RegisterTwoVolumes

g++ -framework OpenCL TransformVolume.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o TransformVolume