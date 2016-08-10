FROM ubuntu:14.04

ENV LD_LIBRARY_PATH /Downloads/BROCCOLI/code/BROCCOLI_LIB/clBLASLinux 
ENV BROCCOLI_DIR /Downloads/BROCCOLI/ 
ENV PATH $PATH:/Downloads/BROCCOLI/compiled/Bash/Linux/Release/ 
ENV PATH $PATH:/Downloads/BROCCOLI/code/bids
ENV FSLDIR /usr/share/fsl/5.0
ENV FSLOUTPUTTYPE NIFTI_GZ
ENV PATH /usr/lib/fsl/5.0:$PATH
ENV FSLMULTIFILEQUIT TRUE
ENV POSSUMDIR /usr/share/fsl/5.0
ENV LD_LIBRARY_PATH /usr/lib/fsl/5.0:$LD_LIBRARY_PATH
ENV FSLTCLSH /usr/bin/tclsh
ENV FSLWISH /usr/bin/wish

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get -f install 

RUN apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y g++ && \
    apt-get install -y xorg && \
    apt-get install -y zlib1g-dev 



RUN mkdir /oasis
RUN mkdir /projects
RUN mkdir /scratch
RUN mkdir /local-scratch


RUN mkdir Downloads && \
    cd Downloads && \
    wget https://dl.dropboxusercontent.com/u/4494604/intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64.tar && \
    tar -xf intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64.tar && \
    cd intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64 && \
    chmod +x install.sh && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/RPM/NONRPM/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    rm intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64.tar && \
    mv /opt/intel/intel-opencl-1.2-6.2.0.1760/opencl-sdk /opt/intel/ && \
    rm -rf intel_sdk_for_opencl_2016_ubuntu_6.2.0.1760_x64


RUN cd /Downloads && \
    wget https://dl.dropboxusercontent.com/u/4494604/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz  && \
    tar -xf opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz && \
    cd opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25 && \
    chmod +x install.sh && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/RPM/NONRPM/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && \
    rm opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz && \
    rm -rf opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25



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
    rm rest_sm.nii.gz


ENTRYPOINT ["/Downloads/BROCCOLI/code/bids/broccolipipeline.sh"]

