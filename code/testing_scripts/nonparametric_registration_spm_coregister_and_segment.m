
clear all
clc
close all

if ispc
    addpath('D:/spm8')
    data_path = 'D:/BROCCOLI_test_data/Cambridge';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';    
elseif isunix
    addpath('/data/andek/spm8/')
    data_path = '/data/andek/BROCCOLI_test_data/Cambridge/wSPM/';
   	results_directory = '/data/andek/BROCCOLI_test_data/SPM/normalization/segment/';
end

try
    system(['rm' ' batch_preprocessing.mat']);
end
   
voxel_size = 2; % 1 or 2
dirs = dir(data_path);

N = 198;    % Number of subjects
normalization_times = zeros(N,1);

% Loop over subjects
tic
for s = 1:N
        
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    spm_jobman('initcfg'); 
        
    subject = dirs(s+2).name    % Skip . and .. 'folders'
    subject_path = [data_path subject '/anat/'];
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(subject_path);      
            
    %% Coregister settings
    
    pjobs{2}.spatial{1}.coreg{1}.estwrite.ref = {['/home/andek/fsl/data/standard/MNI152_T1_' num2str(voxel_size) 'mm_brain_.nii,1']};
    pjobs{2}.spatial{1}.coreg{1}.estwrite.source = {['/data/andek/BROCCOLI_test_data/Cambridge/wSPM/' subject '/anat/mprage_skullstripped.nii,1']};    
    pjobs{2}.spatial{1}.coreg{1}.estwrite.other = {''};
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.cost_fun = 'nmi';
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.sep = [4 2];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.fwhm = [7 7];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.interp = 1;
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.wrap = [0 0 0];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.mask = 0;
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.prefix = 'r';
    
    %% Segment settings        

    pjobs{3}.spatial{1}.preproc.data = {['/data/andek/BROCCOLI_test_data/Cambridge/wSPM/' subject '/anat/rmprage_skullstripped.nii,1']};
    pjobs{3}.spatial{1}.preproc.output.GM = [0 0 1];
    pjobs{3}.spatial{1}.preproc.output.WM = [0 0 1];
    pjobs{3}.spatial{1}.preproc.output.CSF = [0 0 0];
    pjobs{3}.spatial{1}.preproc.output.biascor = 1;
    pjobs{3}.spatial{1}.preproc.output.cleanup = 0;
    pjobs{3}.spatial{1}.preproc.opts.tpm = {
                                               '/home/andek/spm8/tpm/grey.nii'
                                               '/home/andek/spm8/tpm/white.nii'
                                               '/home/andek/spm8/tpm/csf.nii'
                                               };
    pjobs{3}.spatial{1}.preproc.opts.ngaus = [2
                                              2
                                              2
                                              4];
    pjobs{3}.spatial{1}.preproc.opts.regtype = 'mni';
    pjobs{3}.spatial{1}.preproc.opts.warpreg = 1;
    pjobs{3}.spatial{1}.preproc.opts.warpco = 25;
    pjobs{3}.spatial{1}.preproc.opts.biasreg = 0.0001;
    pjobs{3}.spatial{1}.preproc.opts.biasfwhm = 60;
    pjobs{3}.spatial{1}.preproc.opts.samp = 3;
    pjobs{3}.spatial{1}.preproc.opts.msk = {''};


    %% Normalize write settings
    
    % Transform T1 volume using estimated parameters
    pjobs{4}.spatial{1}.normalise{1}.write.subj.matname = {['/data/andek/BROCCOLI_test_data/Cambridge/wSPM/' subject '/anat/rmprage_skullstripped_seg_sn.mat']};
    pjobs{4}.spatial{1}.normalise{1}.write.subj.resample = {['/data/andek/BROCCOLI_test_data/Cambridge/wSPM/' subject '/anat/rmprage_skullstripped.nii,1']};
    pjobs{4}.spatial{1}.normalise{1}.write.roptions.preserve = 0;
    %pjobs{4}.spatial{1}.normalise{1}.write.roptions.bb = [-78 -112 -50
    %                                                      78 76 85];
    % Make bigger bounding box to get same number of voxels as MNI template                                                       
    if (voxel_size == 1)
        pjobs{4}.spatial{1}.normalise{1}.write.roptions.bb = [-91 -126 -72
                                                               90   91 109];
    elseif (voxel_size == 2)
        pjobs{4}.spatial{1}.normalise{1}.write.roptions.bb = [-90 -126 -72
                                                               90   90 108];
    end
    
    pjobs{4}.spatial{1}.normalise{1}.write.roptions.vox = [voxel_size voxel_size voxel_size];
    pjobs{4}.spatial{1}.normalise{1}.write.roptions.interp = 1;
    pjobs{4}.spatial{1}.normalise{1}.write.roptions.wrap = [0 0 0];
    pjobs{4}.spatial{1}.normalise{1}.write.roptions.prefix = 'w';
    
    save('batch_preprocessing.mat','pjobs');
    
    
    error1 = 0;
    start = clock;
    try        
        % Run processing
        spm_jobman('run',pjobs);        
    catch err
        err
        error1 = 1;
    end
    normalization_times(s) = etime(clock,start);
    
        
    coregistered_data = ['rmprage_skullstripped.nii'];
    normalized_data = ['wrmprage_skullstripped.nii'];
    new_normalized_data = ['SPM_warped_subject_' num2str(s) '.nii'];
    mat1 = 'rmprage_skullstripped_seg_sn.mat';
    mat2 = 'rmprage_skullstripped_seg_inv_sn.mat';
    c1 = 'c1rmprage_skullstripped.nii';
    c2 = 'c2rmprage_skullstripped.nii';
    m = 'mrmprage_skullstripped.nii';
    
    % Move files to results directory 
    system(['mv ' normalized_data ' ' results_directory new_normalized_data]);
    
    % Remove temporary data
    system(['rm ' coregistered_data]);
    system(['rm ' mat1]);
    system(['rm ' mat2]);
    system(['rm ' c1]);
    system(['rm ' c2]);
    system(['rm ' m]);
    system(['rm' ' batch_preprocessing.mat']);
    
    s
        
    try
        system(['rm' ' batch_preprocessing.mat']);
    end
    
end
toc







