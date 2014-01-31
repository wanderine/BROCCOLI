%  	 BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
%    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

if ispc
    addpath('D:\nifti_matlab')
    basepath = 'D:\OpenfMRI\';
    opencl_platform = 0; % 0 Nvidia, 1 Intel, 2 AMD
    opencl_device = 0;
    
    %mex -g GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 2;
    opencl_device = 0;
    
    %mex -g GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'RhymeJudgment/ds003_models';
number_of_subjects = 49;

voxel_size = 2;
do_permutations_in_Matlab = 0;

%--------------------------------------------------------------------------------------
% Statistical settings
%--------------------------------------------------------------------------------------

number_of_regressors = 2;

mytimes = zeros(25,1);
for number_of_regressors = 1:1
    
number_of_permutations = 100;
inference_mode = 1; % 0 = voxel, 1 = cluster extent, 2 = cluster mass
cluster_defining_threshold = 3;

%--------------------------------------------------------------------------------------
% Load MNI templates
%--------------------------------------------------------------------------------------

MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

MNI_brain_mask(:,:,1) = 0;
MNI_brain_mask(:,:,end) = 0;

[MNI_sy MNI_sx MNI_sz] = size(MNI_brain_mask);
[MNI_sy MNI_sx MNI_sz]

%--------------------------------------------------------------------------------------
% Load first level results
%--------------------------------------------------------------------------------------

first_level_results = zeros(MNI_sy,MNI_sx,MNI_sz,number_of_subjects);
for subject = 1:13
    if subject < 10
        beta_volume = load_nii([basepath study '/sub00' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    else
        beta_volume = load_nii([basepath study '/sub0' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    end
    beta_volume = double(beta_volume.img);
    first_level_results(:,:,:,subject) = beta_volume;
end

% subject = 1;
% beta_volume = load_nii([basepath study '/sub00' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
% beta_volume = double(beta_volume.img);
% for s = 14:number_of_subjects
%     first_level_results(:,:,:,s) = beta_volume;
% end

subject = 1;
for s = 14:number_of_subjects
    if subject < 10
        beta_volume = load_nii([basepath study '/sub00' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    else
        beta_volume = load_nii([basepath study '/sub0' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    end    
    beta_volume = double(beta_volume.img);
    first_level_results(:,:,:,s) = beta_volume;
    subject = subject + 1;
    if subject > 13
        subject = 1;
    end
end


%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

% X_GLM = zeros(number_of_subjects,number_of_regressors);
% 
% for subject = 1:number_of_subjects
%     
%     X_GLM(subject,1) = randn;
%     
%     for r = 2:number_of_regressors
%         X_GLM(subject,r) = randn;
%     end
%     
% end

X_GLM = [

    0.5377    1.1093   -0.4390   -0.5336    1.5270   -1.5651    0.0012    0.7477   -0.5700    0.5455   -0.6169     1.8339   -0.8637   -1.7947   -2.0026    0.4669   -0.0845   -0.0708   -0.2730   -1.0257   -1.0516    0.2748      0.3539   -0.5861   -0.6300
   -2.2588    0.0774    0.8404    0.9642   -0.2097    1.6039   -2.4863    1.5763   -0.9087    0.3975    0.6011     0.8622   -1.2141   -0.8880    0.5201    0.6252    0.0983    0.5812   -0.4809   -0.2099   -0.7519    0.0923 1.5970    0.7449   -0.0469
    0.3188   -1.1135    0.1001   -0.0200    0.1832    0.0414   -2.1924    0.3275   -1.6989    1.5163    1.7298    -1.3077   -0.0068   -0.5445   -0.0348   -1.0298   -0.7342   -2.3193    0.6647    0.6076   -0.0326   -0.6086  0.5275   -0.8282    2.6830
   -0.4336    1.5326    0.3035   -0.7982    0.9492   -0.0308    0.0799    0.0852   -0.1178    1.6360   -0.7371     0.3426   -0.7697   -0.6003    1.0187    0.3071    0.2323   -0.9485    0.8810    0.6992   -0.4251   -1.7499  0.8542    0.5745   -1.1467
    3.5784    0.3714    0.4900   -0.1332    0.1352    0.4264    0.4115    0.3232    0.2696    0.5894    0.9105     2.7694   -0.2256    0.7394   -0.7145    0.5152   -0.3728    0.6770   -0.7841    0.4943   -0.0628    0.8671 1.3418    0.2818    0.5530
   -1.3499    1.1174    1.7119    1.3514    0.2614   -0.2365    0.8577   -1.8054   -1.4831   -2.0220   -0.0799     3.0349   -1.0891   -0.1941   -0.2248   -0.9415    2.0237   -0.6912    1.8586   -1.0203   -0.9821    0.8985 -2.4995    1.1393   -1.0765
    0.7254    0.0326   -2.1384   -0.5890   -0.1623   -2.2584    0.4494   -0.6045   -0.4470    0.6125    0.1837    -0.0631    0.5525   -0.8396   -0.2938   -0.1461    2.2294    0.1006    0.1034    0.1097   -0.0549    0.2908 -0.1676   -0.4259    1.0306
    0.7147    1.1006    1.3546   -0.8479   -0.5320    0.3376    0.8261    0.5632    1.1287   -1.1187    0.1129    -0.2050    1.5442   -1.0722   -1.1201    1.6821    1.0001    0.5362    0.1136   -0.2900   -0.6264    0.4400 0.3530    0.6361    0.3275
   -0.1241    0.0859    0.9610    2.5260   -0.8757   -1.6642    0.8979   -0.9047    1.2616    0.2495    0.1017     1.4897   -1.4916    0.1240    1.6555   -0.4838   -0.5900   -0.1319   -0.4677    0.4754   -0.9930    2.7873 0.7173    0.7932    0.6521
    1.4090   -0.7423    1.4367    0.3075   -0.7120   -0.2781   -0.1472   -0.1249    1.1741    0.9750   -1.1667     1.4172   -1.0616   -1.9609   -1.2571   -1.1742    0.4227    1.0078    1.4790    0.1269   -0.6407   -1.8543 -1.3049   -0.8984   -0.2789
    0.6715    2.3505   -0.1977   -0.8655   -0.1922   -1.6702   -2.1237   -0.8608   -0.6568    1.8089   -1.1407    -1.2075   -0.6156   -1.2078   -0.1765   -0.2741    0.4716   -0.5046    0.7847   -1.4814   -1.0799   -1.0933 -1.0059    0.1562    0.2452
    0.7172    0.7481    2.9080    0.7914    1.5301   -1.2128   -1.2706    0.3086    0.1555    0.1992   -0.4336     1.6302   -0.1924    0.8252   -1.3320   -0.2490    0.0662   -0.3826   -0.2339    0.8186   -1.5210   -0.1685 0.7907    1.5973    1.4725
    0.4889    0.8886    1.3790   -2.3299   -1.0642    0.6524    0.6487   -1.0570   -0.2926   -0.7236   -0.2185     1.0347   -0.7648   -1.0582   -1.4491    1.6035    0.3271    0.8257   -0.2841   -0.5408   -0.5933    0.5413  -0.1166    0.1124   -2.2751
    0.7269   -1.4023   -0.4686    0.3335    1.2347    1.0826   -1.0149   -0.0867   -0.3086    0.4013    0.3893    -0.3034   -1.4224   -0.2725    0.3914   -0.2296    1.0061   -0.4711   -1.4694   -1.0966    0.9421    0.7512 0.5531   -0.3086   -1.6333
    0.2939    0.4882    1.0984    0.4517   -1.5062   -0.6509    0.1370    0.1922   -0.4930    0.3005    1.7783    -0.7873   -0.1774   -0.2779   -0.1303   -0.4446    0.2571   -0.2919   -0.8223   -0.1807   -0.3731    1.2231 -0.9606    0.4567    0.4155
    0.8884   -0.1961    0.7015    0.1837   -0.1559   -0.9444    0.3018   -0.0942    0.0458    0.8155   -1.2833    -1.1471    1.4193   -2.0518   -0.4762    0.2761   -1.3218    0.3999    0.3362   -0.0638    0.7989   -2.3290 -1.6338   -0.2751   -0.6548
   -1.0689    0.2916   -0.3538    0.8620   -0.2612    0.9248   -0.9300   -0.9047    0.6113    0.1202    0.9019    -0.8095    0.1978   -0.8236   -1.3617    0.4434    0.0000   -0.1768   -0.2883    0.1093    0.5712   -1.8356 0.7612    0.4431   -0.2963
   -2.9443    1.5877   -1.5771    0.4550    0.3919   -0.0549   -2.1321    0.3501    1.8140    0.4128    0.0668     1.4384   -0.8045    0.5080   -0.8487   -1.2507    0.9111    1.1454   -1.8359    0.3120   -0.9870    0.0355 1.1933   -0.1348   -1.4969
    0.3252    0.6966    0.2820   -0.3349   -0.9480    0.5946   -0.6291    1.0360    1.8045    0.7596    2.2272    -0.7549    0.8351    0.0335    0.5528   -0.7411    0.3502   -1.2038    2.4245   -0.7231   -0.6572   -0.0692 1.6321   -0.0183   -0.9048
    1.3703   -0.2437   -1.3337    1.0391   -0.5078    1.2503   -0.2539    0.9594    0.5265   -0.6039   -0.5073    -1.7115    0.2157    1.1275   -1.1176   -0.3206    0.9298   -1.4286   -0.3158   -0.2603    0.1769    0.2358 -1.5322    0.4608   -0.4042
   -0.1022   -1.1658    0.3502    1.2607    0.0125    0.2398   -0.0209    0.4286    0.6001   -0.3075    0.2458    -0.2414   -1.1480   -0.2991    0.6601   -3.0292   -0.6904   -0.5607   -1.0360    0.5939   -0.1318    0.0700  -1.3369    1.3623   -0.7258
    0.3192    0.1049    0.0229   -0.0679   -0.4570   -0.6516    2.1778    1.8779   -2.1860    0.5954   -0.6086     0.3129    0.7223   -0.2620   -0.1952    1.2424    1.1921    1.1385    0.9407   -1.3270    1.0468   -1.2226  -1.4738    0.4519   -0.8665
   -0.8649    2.5855   -1.7502   -0.2176   -1.0667   -1.6118   -2.4969    0.7873   -1.4410   -0.1980    0.3165    -0.0301   -0.6669   -0.2857   -0.3031    0.9337   -0.0245    0.4413   -0.8759    0.4018    0.3277   -1.3429 -0.0417    1.6484   -0.4218
   -0.1649    0.1873   -0.8314    0.0230    0.3503   -1.9488   -1.3981    0.3199    1.4702   -0.2383   -1.0322     0.6277   -0.0825   -0.9792    0.0513   -0.0290    1.0205   -0.2551   -0.5583   -0.3268    0.2296    1.3312 -0.6155   -2.0284   -0.9427
    1.0933   -1.9330   -1.1564    0.8261    0.1825    0.8617    0.1644   -0.3114    0.8123    0.4400   -0.4189    -0.1403   -1.1555   -1.8963   -0.1609   -1.4264    0.4980    0.6395   -1.1464   -0.6936   -1.5566   -1.4424  1.3142   -0.4493    1.3419
    0.8998   -0.0095   -2.1280    0.4093   -1.0145    2.7891   -0.0810    0.5476    1.2815    1.9151    1.3025    -0.3001   -0.6898   -1.1769   -0.9526   -0.2133    0.7276    0.5409    1.5651   -0.8097    0.6098    1.4099 -1.4551    0.2360   -0.9884
    1.0294   -0.6667   -0.9905    0.3173   -0.3253   -0.7731   -1.2626   -1.6933   -1.2368   -0.6479   -1.6625    -0.3451    0.8641   -1.1730    0.0780    1.9444    0.8366    1.1104   -0.4494    0.2147    2.6173    1.9437 -1.7423   -0.8352    1.8179
    1.0128    0.1134   -1.7254    1.3244   -0.5718   -1.1283   -0.9896   -0.0843    2.0108    0.5510   -1.0847     0.6293    0.3984    0.2882   -0.2132   -0.2500   -1.4245   -1.8288   -1.9920    0.0256    0.2942    0.2268 0.2053   -1.2760   -0.3744
   -0.2130    0.8840   -1.5942   -0.1345   -1.5693    0.7174    1.3845    0.8412    0.3083   -0.7778    1.0989    -0.8657    0.1803    0.1102   -1.1714   -0.4774   -0.7779   -0.0627   -0.4147   -0.9382   -1.0649    0.1472 1.1929    0.6170   -1.4517
   -1.0431    0.5509    0.7871   -1.3853   -1.3380    0.3160    0.4489    1.9122    1.6742   -1.7684    2.2957    -0.2701    0.6830   -0.0022    0.3105    0.0303    1.4065   -0.3633   -0.3909    0.1250   -0.4229    2.7526 -0.8028    0.6127   -0.6187
   -0.4381    1.1706    0.0931   -0.2495    0.8531    0.4011   -1.0206    0.4092    0.5301   -1.0531    0.1383    -0.4087    0.4759   -0.3782    0.5037    0.4043    0.9297   -3.0730   -1.1424   -0.9521    0.6478   -1.9071 -1.2656    0.2894    0.9345
    0.9835    1.4122   -1.4827   -0.8927   -0.7006   -1.6058    0.6263   -0.6249    0.8540   -0.3176   -0.3650    -0.2977    0.0226   -0.0438    1.9085   -1.6305    0.6615   -0.2867   -1.1687    0.3891    1.7690   -0.8481 -0.1493    0.3953    1.0559
    1.1437   -0.0479    0.9608    0.1222    1.4600    2.1385   -0.1973    0.3926   -1.1560    1.5106   -0.7648    -0.5316    1.7013    1.7382    1.0470    2.0500    0.5411    0.4056    1.3018    0.0397    0.1640   -1.1277 -1.6364   -0.8706    0.1602
    0.9726   -0.5097   -0.4302   -0.2269    0.1205   -1.5409   -1.4193   -0.5936   -0.4506   -0.2828    0.0782    -0.5223   -0.0029   -1.6273   -0.1625   -0.9899   -0.2031   -0.7294    0.4364    0.1092    1.1522    2.1066 0.0173   -0.4977    0.2874
    0.1766    0.9199    0.1663    0.6901    1.1978   -0.5000    1.1473   -0.5044   -0.2506   -1.1465   -0.7158     0.9707    0.1498    0.3763    0.5558   -0.5927    0.3830    0.5979    0.1021   -0.1899    0.6737   -0.2805 0.8284   -0.1067    0.6329
   -0.4140    1.4049   -0.2270   -1.1203   -0.4698    0.4120   -1.2813    1.1963   -1.0329   -0.6691    1.1665    -0.4383    1.0341   -1.1489   -1.5327    0.8864    0.4055   -2.2033    0.1203   -0.3233   -0.4003    1.2128 0.2177   -0.6878   -1.4590
    2.0034    0.2916    2.0243   -1.0979   -1.3852   -0.3638   -0.5712   -1.0368    0.7665   -0.6718    0.4855     0.9510   -0.7777   -2.3595   -1.4158   -1.9568   -0.5993    0.2140   -0.8571    1.7447    0.5756    1.0260 -1.9092    0.3319   -0.5817
   -0.4320    0.5667   -0.5100    0.0596    0.4207   -0.5896    0.9424   -0.1699   -1.1605   -0.7781    0.8707     0.6489   -1.3826   -1.3216   -0.4113    0.4007    0.8535    0.0937   -0.1917    2.3774   -1.0636   -0.3818 -0.5368    2.3652   -1.8301
   -0.3601    0.2445   -0.6361   -0.3680    0.0951   -1.8530   -1.1223   -0.8658    1.5261    0.5530    0.4289     0.7059    0.8084    0.3179   -1.3610    0.4967   -0.2073    0.3062    0.1807    0.1685   -0.4234   -0.2991 -0.3020   -0.4822   -0.4491
    1.4158    0.2130    0.1380    0.7796    1.0822    0.2704   -1.1723    1.2665   -0.3012    0.3616   -0.8999    -1.6045    0.8797   -0.7107    0.4394    0.9704   -0.6528   -0.9610   -0.2512   -0.6987   -0.3519    0.6347 1.8136    0.6474    0.9493
    1.0289    2.0389    0.7770   -0.0896   -0.5686    0.4772   -0.6537   -0.2046    0.8328    0.2695    0.0675     1.4580    0.9239    0.6224    1.0212    0.8100   -0.0713   -1.2294   -2.2015   -0.6946   -2.5644   -0.1871  0.9149   -1.0344    0.7174
    0.0475    0.2669    0.6474   -0.8740    0.1732   -0.9383   -0.2710   -0.7745   -0.4619    0.4659    0.2917     1.7463    0.6417   -0.4256    0.4147   -0.5055    0.1614   -0.9000   -1.3933    0.8836    1.8536    0.9877 -0.0571    1.3396    2.2878
    0.1554    0.4255    1.0486    0.3484   -1.1933   -0.2682   -0.2857   -0.3862    0.4359    1.0393    0.3929    -1.2371   -1.3147    0.6607    0.3493    0.6470   -0.4099   -0.4624    0.5256    0.8967    0.9109    0.1946 1.3094   -0.9691    0.1667
   -2.1935   -0.4164    2.5088   -0.7292   -0.3536   -0.7113   -0.4098    1.5233    0.5047   -0.2397    0.2798    -0.3334    1.2247    1.0635    0.3268    0.0464    0.0614   -0.5035    1.7985   -0.4009    0.1810    0.0512 -1.0447    0.2087   -2.1565
    0.7135   -0.0436    1.1569   -0.5149   -0.7929   -1.8461    1.2333   -0.1169   -0.5138    0.2442   -0.7745     0.3174    0.5824    0.0530   -0.8964   -1.5505   -0.3983    0.6103   -0.3202    0.7964    0.0964    0.7868 -0.3483   -0.6186    1.6894
    0.4136   -1.0065   -1.2884   -1.2033    0.1716   -0.5435    0.0591    0.8175   -0.6712   -0.8305    1.4089    -0.5771    0.0645   -0.3712    1.0378   -0.0621   -0.9119   -1.4669    0.4902    1.1867   -0.3523   -0.5341 1.4126    0.5120    1.2823
    0.1440    0.6003   -0.7578   -0.8459    1.1990    0.6527   -1.6258    0.7653    0.7907   -0.1748    1.9278    -1.6387   -1.3615   -0.5640   -0.1729    0.8017   -0.7343   -1.9648    0.7783    0.2877   -0.4807   -0.1762  1.5024    0.0114   -0.5826
   -0.7601    0.3476    0.5551   -1.2087    1.0533    0.5406    2.6052   -1.4803    0.0032    0.8368   -0.2438    -0.8188   -0.1818   -0.5568   -0.2971   -0.7489    0.9758    0.9724    0.5404    0.3656    2.5383   -0.8976 0.7304   -0.0440    0.2226
    0.5197   -0.9395   -0.8951   -3.2320   -0.9363   -0.1569    0.2570   -0.0915    3.5267   -1.3233   -0.7923    -0.0142   -0.0375   -0.4093   -1.0870   -1.2691    0.2778   -0.9742   -0.7603   -0.1124    0.1283   -0.9530 0.4908    2.9491    0.7795
];

X_GLM = X_GLM(:,1:number_of_regressors);

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';


%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

contrasts = [1 zeros(1,number_of_regressors-1)];

for i = 1:size(contrasts,1)
    contrast = contrasts(i,:)';
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end

%--------------------------------------------------------------------------------------
% Generate permutation matrix
%--------------------------------------------------------------------------------------

permutation_matrix = zeros(number_of_permutations,number_of_subjects);
values = 1:number_of_subjects;
for p = 1:number_of_permutations
    permutation = randperm(number_of_subjects);
    permutation_matrix(p,:) = values(permutation);
end

%permutation_matrix = [5.000000 29.000000 12.000000 16.000000 25.000000 36.000000 18.000000 37.000000 27.000000 49.000000 34.000000 40.000000 20.000000 3.000000 48.000000 42.000000 26.000000 19.000000 33.000000 41.000000 6.000000 22.000000 8.000000 13.000000 15.000000 43.000000 28.000000 7.000000 46.000000 45.000000 31.000000 39.000000 14.000000 38.000000 4.000000 17.000000 30.000000 44.000000 10.000000 23.000000 9.000000 24.000000 21.000000 35.000000 2.000000 11.000000 32.000000 1.000000 47.000000]; 
%permutation_matrix = [28.000000 18.000000 19.000000 23.000000 40.000000 34.000000 41.000000 16.000000 24.000000 3.000000 7.000000 39.000000 13.000000 5.000000 45.000000 2.000000 29.000000 30.000000 44.000000 36.000000 14.000000 32.000000 37.000000 42.000000 11.000000 48.000000 21.000000 20.000000 35.000000 46.000000 22.000000 8.000000 15.000000 31.000000 33.000000 26.000000 49.000000 6.000000 4.000000 10.000000 17.000000 12.000000 43.000000 47.000000 38.000000 9.000000 25.000000 27.000000 1.000000]; 

%permutation_matrix = 1:49;



%-----------------------------------------------------------------------
% Run permutation based second level analysis in Matlab
%-----------------------------------------------------------------------

[sy sx sz st] = size(first_level_results);

statistical_maps_cpu = zeros(sy,sx,sz);
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
residuals_cpu = zeros(sy,sx,sz,st);
residual_variances_cpu = zeros(sy,sx,sz);

disp('Calculating statistical maps')

null_distribution_cpu = zeros(number_of_permutations,1);

if do_permutations_in_Matlab == 1
    
    tic
    % Loop over permutations
    for p = 1:number_of_permutations
        
        statistical_maps_cpu = zeros(sy,sx,sz);
        
        % Loop over voxels
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if MNI_brain_mask(y,x,z) == 1
                        
                        % Calculate beta values, using permuted model
                        data = squeeze(first_level_results(y,x,z,:));
                        permuted_xtxxt_GLM = zeros(size(xtxxt_GLM));
                        permutation = permutation_matrix(p,:);
                        for r = 1:number_of_regressors
                            permuted_xtxxt_GLM(r,:) = xtxxt_GLM(r,permutation);
                        end
                        beta = permuted_xtxxt_GLM * data;
                        betas_cpu(y,x,z,:) = beta;
                        
                        % Calculate t-values and residuals, using permuted model
                        permuted_X_GLM = zeros(size(X_GLM));
                        permutation = permutation_matrix(p,:);
                        for r = 1:number_of_regressors
                            permuted_X_GLM(:,r) = X_GLM(permutation,r);
                        end
                        residuals = data - permuted_X_GLM*beta;
                        residuals_cpu(y,x,z,:) = residuals;
                        %residual_variances_cpu(y,x,z) = sum((residuals-mean(residuals)).^2)/(st - 1);
                        residual_variances_cpu(y,x,z) = sum((residuals).^2)/(st - 1);
                        
                        %t-tests
                        for i = 1:size(contrasts,1)
                            contrast = contrasts(i,:)';
                            statistical_maps_cpu(y,x,z,i) = contrast'*beta / sqrt( residual_variances_cpu(y,x,z) * ctxtxc_GLM(i));                            
                        end
                        
                    end
                end
            end
        end
        
        % Voxel
        if (inference_mode == 0)
            null_distribution_cpu(p) = max(statistical_maps_cpu(:));
        % Cluster extent
        elseif (inference_mode == 1)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
            
            cluster_extents = zeros(N,1);
            for i = 1:N
                cluster_extents(i) = sum(labels(:) == i);
            end
            null_distribution_cpu(p) = max(cluster_extents);
        % Cluster mass
        elseif (inference_mode == 2)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
            
            cluster_masses = zeros(N,1);
            for i = 1:N
                cluster_masses(i) = sum(a(labels(:) == i));
            end
            null_distribution_cpu(p) = max(cluster_masses);
        end
    end
    toc
end


%--------------------------------------------------------------------------------------
% Run permutation based second level analysis with OpenCL
%--------------------------------------------------------------------------------------

permutation_matrix = permutation_matrix - 1;

start = clock;
[beta_volumes, residuals, residual_variances, statistical_maps_opencl, cluster_indices, null_distribution_opencl, permuted_first_level_results] = ...
    GLMTTest_SecondLevel_Permutation(first_level_results,MNI_brain_mask, X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM, uint16(permutation_matrix'), number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device);
elapsed_time = etime(clock,start)

mytimes(number_of_regressors) = elapsed_time;

s = sort(null_distribution_opencl);
threshold_opencl = s(round(0.95*number_of_permutations))

end


slice = round(0.5*MNI_sz);

figure
imagesc(beta_volumes(:,:,slice)); colormap gray; colorbar
title('Beta')

figure
imagesc(statistical_maps_opencl(:,:,slice,1)); colorbar
title('t-values')

figure
imagesc(statistical_maps_opencl(:,:,slice,1) > cluster_defining_threshold); colorbar
title('t-values above cluster defining threshold')

figure
imagesc(cluster_indices(:,:,slice,1)); colorbar
title('Cluster indices')


s = sort(null_distribution_cpu);
threshold_cpu = s(round(0.95*number_of_permutations))

s = sort(null_distribution_opencl);
threshold_opencl = s(round(0.95*number_of_permutations))

if number_of_permutations >= 100    
    figure
    hist(null_distribution_opencl,50)
end

if number_of_permutations < 10
    [null_distribution_cpu  null_distribution_opencl]
end

% Compare clustering to Matlab

% Cluster extent
if (inference_mode == 1)
    
    a = statistical_maps_opencl(:,:,:,1);
    [labels,N] = bwlabeln(a > cluster_defining_threshold);
    
    matlab_sums = zeros(N,1);
    for i = 1:N
        matlab_sums(i) = sum(labels(:) == i);
    end
    
    N = max(cluster_indices(:));
    for i = 1:N
        broccoli_sums(i) = sum(cluster_indices(:) == i);
    end
    
    matlab_sums = sort(matlab_sums);
    broccoli_sums = sort(broccoli_sums);
    
    %[matlab_sums'; broccoli_sums]'
    
    cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
    
% Cluster mass
elseif (inference_mode == 2)
    
    a = statistical_maps_opencl(:,:,:,1);
    [labels,N] = bwlabeln(a > cluster_defining_threshold);
    
    matlab_sums = zeros(N,1);
    for i = 1:N
        matlab_sums(i) = sum(a(labels(:) == i));
    end
    
    N = max(cluster_indices(:));
    for i = 1:N
        broccoli_sums(i) = sum(a(cluster_indices(:) == i));
    end
    
    matlab_sums = sort(matlab_sums);
    broccoli_sums = sort(broccoli_sums);
    
    %[matlab_sums'; broccoli_sums]'
    
    cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
    
end


volume = load_nii(['/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/testing_scripts/randomise/permtest_tstat1.nii.gz']);
volume = double(volume.img);

imagesc([ statistical_maps_cpu(:,:,50) - volume(:,:,50)  ]); colorbar

imagesc([ statistical_maps_cpu(:,:,50) - statistical_maps_opencl(:,:,50)  ]); colorbar






