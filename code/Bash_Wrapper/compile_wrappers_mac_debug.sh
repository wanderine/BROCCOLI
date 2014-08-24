#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_HEADER_DIRECTORY=/System/Library/Frameworks/OpenCL.framework/Headers

# Fist run make for Nifti library
cd nifticlib-2.0.0
make
cd ..

# Compile each wrapper
g++ -g -framework OpenCL  GetOpenCLInfo.cpp -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY}  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug  -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -o GetOpenCLInfo

g++ -g -framework OpenCL MotionCorrection.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o MotionCorrection

g++ -g -framework OpenCL RegisterTwoVolumes.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o RegisterTwoVolumes

g++ -g -framework OpenCL TransformVolume.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o TransformVolume

g++ -g -framework OpenCL RandomiseGroupLevel.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o TransformVolume

g++ -g -framework OpenCL FirstLevelAnalysis.cpp -lBROCCOLI_LIB -lniftiio -lznz -lz -I${OPENCL_HEADER_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Compiled/Mac/Debug -L${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/lib -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o FirstLevelAnalysis


# Now compile the OpenCL kernel code using openclc
/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -emit-llvm-bc broccoli_lib_kernel.cl -o broccoli_lib_kernel.bin


