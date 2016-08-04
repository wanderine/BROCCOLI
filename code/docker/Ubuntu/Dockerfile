FROM ubuntu:16.04

# Enable NeuroDebian repository (if needed)
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
       NEURODEBIAN_ENABLE=yes NEURODEBIAN_MIRROR=origin NEURODEBIAN_FLAVOR=full NEURODEBIAN_UPDATE=0 \
       apt-get install -y neurodebian

# Run apt-get calls
RUN apt-get update \
    && apt-get install -y fsl-5.0-core

ENV FSLDIR=/usr/share/fsl/5.0
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH=/usr/lib/fsl/5.0:$PATH
ENV FSLMULTIFILEQUIT=TRUE
ENV POSSUMDIR=/usr/share/fsl/5.0
ENV LD_LIBRARY_PATH=/usr/lib/fsl/5.0
ENV FSLTCLSH=/usr/bin/tclsh
ENV FSLWISH=/usr/bin/wish

RUN mkdir /oasis
RUN mkdir /projects
RUN mkdir /scratch
RUN mkdir /local-scratch

RUN mkdir Downloads && \
    cd Downloads && \
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



