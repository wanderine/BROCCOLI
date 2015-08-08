#!/bin/bash

BROCCOLI_GIT_DIRECTORY=`git rev-parse --show-toplevel`

cd $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB

# Change release to debug for library
sed -i '' 's/COMPILATION=$RELEASE/COMPILATION=$DEBUG/g' $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB/compile_broccoli_library_mac.sh

# Compile library
./compile_broccoli_library_mac.sh

cd $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper

# Change release to debug for wrappers
sed -i '' 's/COMPILATION=$RELEASE/COMPILATION=$DEBUG/g' $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper/compile_wrappers_mac.sh

# Compile wrappers
./compile_wrappers_mac.sh

# Add compiled debug files
git add $BROCCOLI_GIT_DIRECTORY/compiled/BROCCOLI_LIB/Mac/Debug/libBROCCOLI_LIB.a

git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/FirstLevelAnalysis
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/MotionCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/RegisterTwoVolumes
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/RandomiseGroupLevel
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/SliceTimingCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/TransformVolume
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/GetOpenCLInfo
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/GetBandwidth
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/Smoothing
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/GLM
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Debug/ICA



cd $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB

# Change debug to release
sed -i '' 's/COMPILATION=$DEBUG/COMPILATION=$RELEASE/g' $BROCCOLI_GIT_DIRECTORY/code/BROCCOLI_LIB/compile_broccoli_library_mac.sh

# Compile library
./compile_broccoli_library_mac.sh

cd $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper

# Change debug to release for wrappers
sed -i '' 's/COMPILATION=$DEBUG/COMPILATION=$RELEASE/g' $BROCCOLI_GIT_DIRECTORY/code/Bash_Wrapper/compile_wrappers_mac.sh

# Compile wrappers
./compile_wrappers_mac.sh

# Add compiled release files

git add $BROCCOLI_GIT_DIRECTORY/compiled/BROCCOLI_LIB/Mac/Release/libBROCCOLI_LIB.a

git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/FirstLevelAnalysis
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/MotionCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/RegisterTwoVolumes
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/RandomiseGroupLevel
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/SliceTimingCorrection
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/TransformVolume
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/GetOpenCLInfo
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/GetBandwidth
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/Smoothing
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/GLM
git add $BROCCOLI_GIT_DIRECTORY/compiled/Bash/Mac/Release/ICA





