FROM centos:6.8

RUN rm /bin/sh && \
    ln -s /bin/bash /bin/sh && \
    yum -y install wget && \
    yum -y install git && \
    yum -y install zlib-devel && \
    yum -y install cifs-utils && \
    yum -y install emacs && \
    yum -y install libgomp.x86_64 && \
    yum -y install numactl.x86_64 && \
    yum -y install libXp.x86_64 && \
    yum -y install gcc-c++ && \
    yum -y install libXmu.x86_64 && \
    mkdir Downloads && \
    cd Downloads && \
    wget https://dl.dropboxusercontent.com/u/4494604/opencl_runtime_15.1_x64_5.0.0.57.tar && \
    tar -xf opencl_runtime_15.1_x64_5.0.0.57.tar && \
    cd opencl_runtime_15.1_x64_5.0.0.57 && \
    chmod +x install.sh && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    wget https://dl.dropboxusercontent.com/u/4494604/intel_code_builder_for_opencl_2015_5.0.0.62_x64.tar && \
    tar -xf intel_code_builder_for_opencl_2015_5.0.0.62_x64.tar && \
    cd intel_code_builder_for_opencl_2015_5.0.0.62_x64 && \
    sed -i 's/decline/accept/g' silent.cfg && \
    chmod +x install.sh && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    wget https://github.com/wanderine/BROCCOLI/archive/v1.3.tar.gz && \
    tar -xf v1.3.tar.gz && \
    cd BROCCOLI-1.3 && \
    cd code && \
    cd BROCCOLI_LIB && \
    sed -i '/BROCCOLI/a BROCCOLI_GIT_DIRECTORY=\/Downloads\/BROCCOLI-1.3' compile_broccoli_library.sh && \  
    ./compile_broccoli_library.sh && \  
    cd .. && \
    cd Bash_Wrapper && \
    sed -i '/BROCCOLI/a BROCCOLI_GIT_DIRECTORY=\/Downloads\/BROCCOLI-1.3' compile_wrappers.sh && \
    ./compile_wrappers.sh && \
    cd .. && \
    cd .. && \
    cd compiled/Bash/Linux/Release/ && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Downloads/BROCCOLI-1.3/code/BROCCOLI_LIB/clBLASLinux && \
    export BROCCOLI_DIR=/Downloads/BROCCOLI-1.3/ && \
    ./GetOpenCLInfo && \
    cp /Downloads/BROCCOLI-1.3/test_data/fcon1000/classic/Beijing/sub00440/func/rest.nii.gz . && \
    ./Smoothing rest.nii.gz -verbose && \
    ./MotionCorrection rest.nii.gz 



