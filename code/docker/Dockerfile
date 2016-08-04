FROM centos:6.8

RUN mkdir /oasis

RUN mkdir /projects
 	
RUN mkdir /scratch

RUN mkdir /local-scratch

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
    yum -y install bc-1.06.95-1.el6.x86_64 && \
    yum -y update && \
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
    mkdir BROCCOLI && \
    cd BROCCOLI && \
    git clone --depth 1 https://github.com/wanderine/BROCCOLI.git . && \
    cd code && \
    cd BROCCOLI_LIB && \
    ./compile_broccoli_library.sh && \  
    cd .. && \
    cd Bash_Wrapper && \
    ./compile_wrappers.sh && \
    cd .. && \
    cd .. && \
    cd compiled/Bash/Linux/Release/ && \
    ENV LD_LIBRARY_PATH /Downloads/BROCCOLI/code/BROCCOLI_LIB/clBLASLinux && \
    ENV BROCCOLI_DIR /Downloads/BROCCOLI/ && \
    ENV PATH $PATH:/Downloads/BROCCOLI/compiled/Bash/Linux/Release/ && \
    ./GetOpenCLInfo && \
    cp /Downloads/BROCCOLI/test_data/fcon1000/classic/Beijing/sub00440/func/rest.nii.gz . && \
    ./Smoothing rest.nii.gz -verbose && \
    cd /Downloads/BROCCOLI/code/bids && \
    chmod +x fslinstaller.py && \
    ENV SHELL /bin/bash && \
    python2.6 fslinstaller.py -q -d /usr/local && \
    ENV FSLDIR /usr/local/fsl && \
    . ${FSLDIR}/etc/fslconf/fsl.sh && \
    ENV PATH $PATH:/usr/local/fsl/bin 
    





