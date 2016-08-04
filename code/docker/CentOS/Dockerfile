FROM centos:6.8

ENV LD_LIBRARY_PATH /Downloads/BROCCOLI/code/BROCCOLI_LIB/clBLASLinux 
ENV BROCCOLI_DIR /Downloads/BROCCOLI/ 
ENV PATH $PATH:/Downloads/BROCCOLI/compiled/Bash/Linux/Release/ 
ENV SHELL /bin/bash
ENV FSLDIR /usr/local/fsl 
ENV PATH $PATH:/usr/local/fsl/bin 
ENV FSLOUTPUTTYPE NIFTI_GZ

ENV PATH $PATH:/Downloads/BROCCOLI/code/bids


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
    yum -y update 
   
RUN mkdir Downloads && \
    cd Downloads && \
    wget https://dl.dropboxusercontent.com/u/4494604/opencl_runtime_15.1_x64_5.0.0.57.tar && \
    tar -xf opencl_runtime_15.1_x64_5.0.0.57.tar && \
    cd opencl_runtime_15.1_x64_5.0.0.57 && \
    chmod +x install.sh && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    rm opencl_runtime_15.1_x64_5.0.0.57.tar && \
    rm -rf opencl_runtime_15.1_x64_5.0.0.57 

RUN wget https://dl.dropboxusercontent.com/u/4494604/intel_code_builder_for_opencl_2015_5.0.0.62_x64.tar && \
    tar -xf intel_code_builder_for_opencl_2015_5.0.0.62_x64.tar && \
    cd intel_code_builder_for_opencl_2015_5.0.0.62_x64 && \
    sed -i 's/decline/accept/g' silent.cfg && \
    chmod +x install.sh && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    rm intel_code_builder_for_opencl_2015_5.0.0.62_x64.tar && \
    rm -rf intel_code_builder_for_opencl_2015_5.0.0.62_x64 

RUN cd /Downloads && \
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
    ./GetOpenCLInfo && \
    cp /Downloads/BROCCOLI/test_data/fcon1000/classic/Beijing/sub00440/func/rest.nii.gz . && \
    ./Smoothing rest.nii.gz -verbose && \
    rm rest.nii.gz && \
    rm rest_sm.nii.gz && \
    cd /Downloads/BROCCOLI/code/bids && \
    chmod +x broccolipipeline.sh && \
    chmod +x BIDSto3col.sh 

RUN cd /Downloads/BROCCOLI/code/bids && \
    chmod +x fslinstaller.py && \
    python2.6 fslinstaller.py -q -d /usr/local && \
    . /usr/local/fsl/etc/fslconf/fsl.sh 

ENTRYPOINT ["/Downloads/BROCCOLI/code/bids/broccolipipeline.sh"]



   







