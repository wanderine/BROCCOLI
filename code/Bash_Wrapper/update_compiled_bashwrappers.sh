#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

cd $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB

# Change release to debug for library
sed -i 's/COMPILATION=$RELEASE/COMPILATION=$DEBUG/g' $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB/compile_broccoli_library.sh

# Compile library
./compile_broccoli_library.sh

cd $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper

# Change release to debug for wrappers
sed -i 's/COMPILATION=$RELEASE/COMPILATION=$DEBUG/g' $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper/compile_wrappers.sh

# Compile wrappers
./compile_wrappers.sh

# Add compiled debug files
git add $BROCCOLI_GIT_DIRECTORY/compiled/BROCCOLI_LIB/Linux/Debug/libBROCCOLI_LIB.a

git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/FirstLevelAnalysis
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/MotionCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/RegisterTwoVolumes
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/RandomiseGroupLevel
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/SliceTimingCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/TransformVolume
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/GetOpenCLInfo
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/GetBandwidth
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/Smoothing
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/ICA
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Debug/GLM



cd $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB

# Change debug to release
sed -i 's/COMPILATION=$DEBUG/COMPILATION=$RELEASE/g' $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB/compile_broccoli_library.sh

# Compile library
./compile_broccoli_library.sh

cd $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper

# Change debug to release for wrappers
sed -i 's/COMPILATION=$DEBUG/COMPILATION=$RELEASE/g' $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper/compile_wrappers.sh

# Compile wrappers
./compile_wrappers.sh

# Add compiled release files

git add $BROCCOLI_GIT_DIRECTORY/compiled/BROCCOLI_LIB/Linux/Release/libBROCCOLI_LIB.a

git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/FirstLevelAnalysis
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/MotionCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/RegisterTwoVolumes
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/RandomiseGroupLevel
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/SliceTimingCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/TransformVolume
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/GetOpenCLInfo
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/GetBandwidth
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/Smoothing
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/ICA
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Linux/Release/GLM


