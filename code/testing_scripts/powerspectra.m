close all
clear all
clc

cd /home/andek/Research_projects/BROCCOLI/

addpath('/home/andek/Research_projects/nifti_matlab')


studystring = 'Beijing'
TR = 2;
st_ = 221;

%studystring = 'Cambridge'
%TR = 3;
%st = 115;

MEAN_TIMESERIES = zeros(512,1);
mean_timeseries = zeros(st_,1);

numberofsubjects = 27;

for subject = 1:numberofsubjects
    
    subject
    
    subjectString = ['rest_residuals_' studystring '_boxcar10_0mm_' num2str(subject) '_3iterations.nii.gz'];
    %subjectString = ['rest_residuals_' studystring '_boxcar10_6mm_' num2str(subject) '_3iterations_nomotionregression.nii.gz'];
    
    data = load_nii(subjectString);
    data = data.img;
    data = double(data);
    
    [sy sx sz st] = size(data);
    
    brain_voxels = 0;
    MEAN_TIMESERIES_TEMP = zeros(512,1);
    mean_timeseries_temp = zeros(st_,1);
    
    for z = 1:sz
        for y = 1:sy
            for x = 1:sx
                if data(y,x,z,5) ~= 0
                    timeseries = squeeze(data(y,x,z,5:end));
                    if (std(timeseries) ~= 0)                        
                        timeseries = timeseries/(std(timeseries) + eps);
                        mean_timeseries_temp = mean_timeseries_temp + timeseries;
                        TIMESERIES = fft(timeseries,512);
                        MEAN_TIMESERIES_TEMP = MEAN_TIMESERIES_TEMP + ((abs(TIMESERIES)).^2)/length(timeseries);
                        brain_voxels = brain_voxels + 1;
                    end
                end
            end
        end
    end
    
    mean_timeseries_temp = mean_timeseries_temp / brain_voxels;
    mean_timeseries = mean_timeseries + mean_timeseries_temp;
    
    MEAN_TIMESERIES_TEMP = MEAN_TIMESERIES_TEMP / brain_voxels;
    MEAN_TIMESERIES = MEAN_TIMESERIES + MEAN_TIMESERIES_TEMP;
    
end

mean_timeseries = mean_timeseries / numberofsubjects;

MEAN_TIMESERIES = MEAN_TIMESERIES / numberofsubjects;
MEAN_TIMESERIES = fftshift(MEAN_TIMESERIES);

f = linspace(0,(1/TR)/2,257);
figure
plot(f,MEAN_TIMESERIES(256:end),'-g','LineWidth',3)
hold off
xlabel('Frequency (Hz)','FontSize',20)
ylabel('Power','FontSize',20)
legend('Residuals BROCCOLI')
title(['Power spectra of GLM residuals'],'FontSize',20)

set(gca,'XTick',[0:0.05:((1/TR)/2)])
set(gca,'FontSize',20)
axis([0 (1/TR)/2 0 2])



