% This batch script analyses the Face fMRI dataset available from the SPM site:
% http://www.fil.ion.ucl.ac.uk/spm/data/face_rep/face_rep_SPM5.html
% as described in the manual Chapter 29.

% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Guillaume Flandin
% $Id: face_repetition_single_spm5_batch.m 30 2008-05-20 11:16:55Z guillaume $

%% Path containing data
%--------------------------------------------------------------------------

clear all
clc
close all

tic

processing_times = zeros(13,1);

voxel_size = 2; % 1 or 2
  
for subject = [1 2 3 4 5 6 7 8 9 10 11 12 13]
    
    if subject == 1
        subjectstring = '01';
    elseif subject == 2
        subjectstring = '02';
    elseif subject == 3
        subjectstring = '03';
    elseif subject == 4
        subjectstring = '04';
    elseif subject == 5
        subjectstring = '05';
    elseif subject == 6
        subjectstring = '06';
    elseif subject == 7
        subjectstring = '07';
    elseif subject == 8
        subjectstring = '08';
    elseif subject == 9
        subjectstring = '09';
    elseif subject == 10
        subjectstring = '10';
    elseif subject == 11
        subjectstring = '11';
    elseif subject == 12
        subjectstring = '12';
    elseif subject == 13
        subjectstring = '13';
    end
            
    start = clock;
    
    if ispc
        addpath('D:\spm8')        
        data_path = ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\'];
        results_directory = '/data/andek/BROCCOLI_test_data/SPM/';
    elseif isunix
        addpath('/data/andek/spm8/')        
    end
    
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    
    spm_jobman('initcfg'); 
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear jobs
    jobs{1}.util{1}.cdir.directory = cellstr(data_path);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SPATIAL PREPROCESSING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% REALIGN (motion correction)
    %--------------------------------------------------------------------------
    
    % Gives rbold.nii
    filename = [data_path 'BOLD/task001_run001/bold.nii'];
    jobs{2}.spatial{1}.realign{1}.estwrite.data{1} = cellstr(filename);
    
    %% SLICE TIMING CORRECTION
    %--------------------------------------------------------------------------
    
    % Gives arbold.nii
    jobs{3}.temporal{1}.st.scans = {
        {
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,1']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,2']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,3']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,4']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,5']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,6']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,7']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,8']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,9']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,10']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,11']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,12']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,13']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,14']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,15']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,16']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,17']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,18']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,19']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,20']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,21']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,22']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,23']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,24']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,25']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,26']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,27']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,28']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,29']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,30']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,31']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,32']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,33']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,34']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,35']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,36']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,37']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,38']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,39']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,40']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,41']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,42']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,43']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,44']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,45']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,46']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,47']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,48']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,49']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,50']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,51']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,52']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,53']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,54']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,55']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,56']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,57']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,58']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,59']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,60']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,61']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,62']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,63']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,64']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,65']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,66']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,67']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,68']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,69']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,70']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,71']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,72']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,73']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,74']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,75']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,76']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,77']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,78']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,79']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,80']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,81']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,82']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,83']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,84']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,85']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,86']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,87']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,88']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,89']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,90']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,91']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,92']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,93']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,94']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,95']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,96']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,97']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,98']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,99']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,100']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,101']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,102']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,103']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,104']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,105']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,106']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,107']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,108']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,109']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,110']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,111']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,112']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,113']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,114']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,115']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,116']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,117']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,118']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,119']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,120']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,121']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,122']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,123']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,124']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,125']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,126']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,127']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,128']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,129']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,130']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,131']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,132']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,133']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,134']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,135']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,136']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,137']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,138']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,139']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,140']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,141']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,142']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,143']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,144']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,145']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,146']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,147']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,148']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,149']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,150']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,151']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,152']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,153']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,154']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,155']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,156']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,157']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,158']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,159']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\rbold.nii,160']
        }
        }';
    
    jobs{3}.temporal{1}.st.nslices = 33;
    jobs{3}.temporal{1}.st.tr = 2;
    jobs{3}.temporal{1}.st.ta = 2-2/33;
    jobs{3}.temporal{1}.st.so = 33:-1:1;
    jobs{3}.temporal{1}.st.refslice = 17;
    jobs{3}.temporal{1}.st.prefix = 'a';
    
    %% COREGISTRATION, T1 and T1 template (before segment)
    %--------------------------------------------------------------------------
    
    filename = ['D:/spm8/templates/T1.nii'];
    jobs{4}.spatial{1}.coreg{1}.estwrite.ref = {[filename ',1']};
    filename = [data_path 'anatomy/mprage_defaced.nii'];
    jobs{4}.spatial{1}.coreg{1}.estwrite.source = {[filename ',1']};
    
    %% COREGISTRATION, fMRI and T1
    %--------------------------------------------------------------------------
    
    filename = [data_path 'BOLD/task001_run001/meanbold.nii'];
    jobs{5}.spatial{1}.coreg{1}.estimate.ref = {[filename ',1']};
    filename = [data_path 'anatomy/rmprage_defaced.nii'];
    jobs{5}.spatial{1}.coreg{1}.estimate.source = {[filename ',1']};
        
    %% SEGMENT
    %--------------------------------------------------------------------------
    
    filename = [data_path 'anatomy/rmprage_defaced.nii'];
    jobs{6}.spatial{1}.preproc.data = {[filename ',1']};    
    
    %% NORMALIZE (using transformation from segment)
    %--------------------------------------------------------------------------
    
    job = 7;
    matname = [data_path 'anatomy/rmprage_defaced_seg_sn.mat'];
    jobs{job}.spatial{1}.normalise{1}.write.subj.matname  = cellstr(matname);
    filename = [[data_path 'BOLD/task001_run001/arbold.nii']];
    jobs{job}.spatial{1}.normalise{1}.write.subj.resample = cellstr(filename);
    jobs{job}.spatial{1}.normalise{1}.write.roptions.vox  = [voxel_size voxel_size voxel_size];        
    
    %jobs{job}.spatial{1}.normalise{1}.write.roptions.bb = [-78 -112 -50
    %                                                        78   76  85];
    
    % Make bigger bounding box to get same number of mm as before    
    if (voxel_size == 2)
        jobs{job}.spatial{1}.normalise{1}.write.roptions.bb = [-78 -112 -50
                                                                79   77  85];
    elseif (voxel_size == 1)
        jobs{job}.spatial{1}.normalise{1}.write.roptions.bb = [-78 -112 -50
                                                                78   76  85];
    end
    
    jobs{job}.spatial{1}.normalise{2}.write.subj.matname  = cellstr(matname);
    filename = [data_path 'anatomy/mrmprage_defaced.nii'];
    jobs{job}.spatial{1}.normalise{2}.write.subj.resample = cellstr(filename)
    jobs{job}.spatial{1}.normalise{2}.write.roptions.vox  = [voxel_size voxel_size voxel_size];
    
    % Make bigger bounding box to get same number of mm as before
    if (voxel_size == 2)
        jobs{job}.spatial{1}.normalise{2}.write.roptions.bb = [-78 -112 -50
                                                                79   77  85];
    elseif (voxel_size == 1)
        jobs{job}.spatial{1}.normalise{2}.write.roptions.bb = [-78 -112 -50
                                                                78   76  85];
    end
    
    %% SMOOTHING, 6 mm
    %--------------------------------------------------------------------------
    
    job = 8;
    filename = [data_path 'BOLD/task001_run001/warbold.nii'];
    jobs{job}.spatial{1}.smooth.data = cellstr(filename);
    jobs{job}.spatial{1}.smooth.fwhm = [6 6 6];
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% RUN
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save('batch_preprocessing.mat','jobs');
    % %spm_jobman('interactive',jobs);
    spm_jobman('run',jobs);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CLASSICAL STATISTICAL ANALYSIS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    spm_file = ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\classical\SPM.mat'];
    if exist(spm_file,'file')==2
        %system(['rm' spm_file]); % Linux
        delete(spm_file)
    end
    
    clear jobs
    jobs{1}.util{1}.cdir.directory = cellstr(data_path);
    jobs{1}.util{1}.md.basedir = cellstr(data_path);
    jobs{1}.util{1}.md.name = 'classical';
    
    %% MODEL SPECIFICATION AND ESTIMATION
    %--------------------------------------------------------------------------
    data_path = ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\classical\'];
    filename = [data_path ];
    jobs{2}.stats{1}.fmri_spec.dir = cellstr(filename);
    jobs{2}.stats{1}.fmri_spec.timing.units = 'secs';
    jobs{2}.stats{1}.fmri_spec.timing.RT = 2;
    
    scans = {
        {
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,1']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,2']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,3']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,4']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,5']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,6']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,7']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,8']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,9']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,10']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,11']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,12']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,13']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,14']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,15']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,16']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,17']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,18']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,19']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,20']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,21']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,22']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,23']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,24']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,25']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,26']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,27']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,28']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,29']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,30']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,31']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,32']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,33']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,34']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,35']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,36']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,37']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,38']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,39']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,40']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,41']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,42']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,43']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,44']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,45']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,46']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,47']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,48']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,49']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,50']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,51']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,52']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,53']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,54']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,55']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,56']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,57']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,58']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,59']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,60']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,61']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,62']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,63']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,64']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,65']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,66']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,67']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,68']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,69']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,70']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,71']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,72']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,73']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,74']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,75']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,76']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,77']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,78']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,79']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,80']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,81']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,82']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,83']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,84']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,85']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,86']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,87']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,88']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,89']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,90']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,91']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,92']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,93']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,94']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,95']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,96']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,97']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,98']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,99']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,100']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,101']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,102']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,103']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,104']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,105']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,106']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,107']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,108']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,109']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,110']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,111']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,112']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,113']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,114']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,115']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,116']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,117']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,118']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,119']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,120']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,121']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,122']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,123']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,124']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,125']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,126']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,127']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,128']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,129']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,130']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,131']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,132']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,133']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,134']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,135']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,136']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,137']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,138']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,139']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,140']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,141']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,142']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,143']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,144']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,145']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,146']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,147']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,148']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,149']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,150']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,151']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,152']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,153']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,154']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,155']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,156']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,157']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,158']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,159']
        ['D:\OpenfMRI\RhymeJudgment\ds003\sub0' subjectstring '\BOLD\task001_run001\swarbold.nii,160']
        }
        }';
    
    jobs{2}.stats{1}.fmri_spec.sess.scans = scans{1};
    
    jobs{2}.stats{1}.fmri_spec.sess.cond(1).name = 'task1';
    jobs{2}.stats{1}.fmri_spec.sess.cond(1).onset = [20.001 22.501	25.001 27.501 30.001 32.501 35.001 37.501 60.002 62.502	65.002 67.502 70.002 72.502	75.002 77.503 100.003 102.503 105.003 107.503 110.004 112.504 115.004 117.504 140.004 142.504 145.005 147.505 150.005 152.505 155.005 157.505];
    jobs{2}.stats{1}.fmri_spec.sess.cond(1).duration = 2;
    
    jobs{2}.stats{1}.fmri_spec.sess.cond(2).name = 'task2';
    jobs{2}.stats{1}.fmri_spec.sess.cond(2).onset = [180.006 182.506 185.006 187.506 190.006 192.506 195.006 197.507 220.007 222.507 225.007 227.507 230.007 232.508 235.008 237.508 260.008 262.508 265.009 267.509 270.009 272.509 275.009 277.509 300.010 302.510 305.010 307.510 310.010 312.510 315.010 317.510];
    jobs{2}.stats{1}.fmri_spec.sess.cond(2).duration = 2;
    
    % Motion regressors
    jobs{2}.stats{1}.fmri_spec.sess.multi_reg = {['D:/OpenfMRI/RhymeJudgment/ds003/sub0' subjectstring '/' 'BOLD/task001_run001/rp_bold.txt']};
    jobs{2}.stats{1}.fmri_spec.bases.hrf.derivs = [1 0];
    
    filename = [data_path 'SPM.mat'];
    jobs{2}.stats{2}.fmri_est.spmmat = cellstr(filename);
    
    filename = [data_path 'SPM.mat'];
    jobs{2}.stats{3}.con.spmmat = cellstr(filename);
    jobs{2}.stats{3}.con.consess{1}.tcon = struct('name','task1 > rest','convec', 1,'sessrep','none');
    
    filename = [data_path 'SPM.mat'];
    jobs{2}.stats{4}.results.spmmat = cellstr(filename);
    jobs{2}.stats{4}.results.conspec.contrasts = Inf;
    jobs{2}.stats{4}.results.conspec.threshdesc = 'none';
    jobs{2}.stats{4}.results.conspec.thresh = 1e-6; % threshold
    jobs{2}.stats{4}.results.conspec.extent = 0;
    
    save('batch_analysis.mat','jobs');
    spm_jobman('run',jobs);
    
    processing_times(subject) = etime(clock,start);
        
end
toc


