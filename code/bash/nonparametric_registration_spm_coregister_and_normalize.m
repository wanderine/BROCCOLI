
clear all
clc
close all

if ispc
    addpath('D:/spm8')
    data_path = 'D:/BROCCOLI_test_data/Cambridge';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';    
elseif isunix
    addpath('/data/andek/spm8/')
    data_path = '/data/andek/BROCCOLI_test_data/Cambridge/';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/normalize/';
end

try
    system(['rm' ' batch_preprocessing.mat']);
end
    
dirs = dir(data_path);

% Loop over subjects
tic
for s = 1:198
    
    
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    spm_jobman('initcfg'); % useful in SPM8 only
    
    
    subject = dirs(s+2).name    
    subject_path = [data_path subject '/anat/'];
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(subject_path);      
            
    %% Coregister settings
    
    pjobs{2}.spatial{1}.coreg{1}.estwrite.ref = {'/home/andek/fsl/data/standard/MNI152_T1_1mm_brain_.nii,1'};
    pjobs{2}.spatial{1}.coreg{1}.estwrite.source = {['/data/andek/BROCCOLI_test_data/Cambridge/' subject '/anat/mprage_skullstripped.nii,1']};    
    pjobs{2}.spatial{1}.coreg{1}.estwrite.other = {''};
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.cost_fun = 'nmi';
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.sep = [4 2];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.eoptions.fwhm = [7 7];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.interp = 1;
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.wrap = [0 0 0];
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.mask = 0;
    pjobs{2}.spatial{1}.coreg{1}.estwrite.roptions.prefix = 'r';


    %% Normalize settings        
   
    pjobs{3}.spatial{1}.normalise{1}.estwrite.subj.source = {['/data/andek/BROCCOLI_test_data/Cambridge/' subject '/anat/rmprage_skullstripped.nii,1']};    
    pjobs{3}.spatial{1}.normalise{1}.estwrite.subj.wtsrc = '';
    pjobs{3}.spatial{1}.normalise{1}.estwrite.subj.resample = {['/data/andek/BROCCOLI_test_data/Cambridge/' subject '/anat/rmprage_skullstripped.nii']};    
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.template = {'/home/andek/fsl/data/standard/MNI152_T1_1mm_brain_.nii'};
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.weight = '';
    %pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.smosrc = 8;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.smosrc = 0;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.smoref = 0;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.regtype = 'mni';
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.cutoff = 25;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.nits = 16;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.eoptions.reg = 1;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.preserve = 0;
    
    % Standard bounding box
    %pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.bb = [-78 -112 -50
    %                                                         78 76 85];
     
    % Make bigger bounding box to get same number of voxels as MNI template
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.bb = [-91 -126 -72
                                                             90 91 109];
    % Write 1 mm voxels
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.vox = [1 1 1];
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.interp = 1;
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.wrap = [0 0 0];
    pjobs{3}.spatial{1}.normalise{1}.estwrite.roptions.prefix = 'w';


    save('batch_preprocessing.mat','pjobs');
    
    
    error1 = 0;
    try        
        % Run processing
        spm_jobman('run',pjobs);        
    catch err
        err
        error1 = 1;
    end
    
    
    % Move files to results directory 
    coregistered_data = ['rmprage_skullstripped.nii'];
    normalized_data = ['wrmprage_skullstripped.nii'];
    new_normalized_data = ['SPM_warped_subject_' num2str(s) '.nii'];
    mat = 'rmprage_skullstripped_sn.mat';
    
    system(['mv ' normalized_data ' ' results_directory new_normalized_data]);
    system(['rm ' coregistered_data]);
    system(['rm ' mat]);
    system(['rm' ' batch_preprocessing.mat']);
    
    s
        
    try
        system(['rm' ' batch_preprocessing.mat']);
    end
    
end
toc







