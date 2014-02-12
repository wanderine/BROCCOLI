#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`
OPENCL_HEADER_DIRECTORY=/usr/local/cuda-5.0/include/CL
OPENCL_LIBRARY_DIRECTORY=/usr/lib

gcc MotionCorrection.cpp ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c -lOpenCL -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o MotionCorrection 

gcc RegisterTwoVolumes.cpp ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c -lOpenCL -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o RegisterTwoVolumes

gcc TransformVolume.cpp ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c ${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c -lOpenCL -lBROCCOLI_LIB -I${OPENCL_HEADER_DIRECTORY} -L${OPENCL_LIBRARY_DIRECTORY} -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -L${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/ -I${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB/Eigen -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I${BROCCOLI_GIT_DIRECTORY}/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o TransformVolume

#gcc RegisterTwoVolumes.cpp /home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c /home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c  -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/ -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/ -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o RegisterTwoVolumes


#gcc MotionCorrection.cpp /home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/niftilib/nifti1_io.c /home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/znzlib/znzlib.c  -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/ -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/niftilib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/ -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/Bash_Wrapper/nifticlib-2.0.0/znzlib -o MotionCorrection


