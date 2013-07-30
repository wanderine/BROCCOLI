function [compensated_volume,registration_parameters, rotations, scalings] = Phasebased_3D_Registration_FFT(vol1,vol2,quadrature_responses_reference_volume,quadrature_filters,x,y,z,max_its,filter_size)

q11 = quadrature_responses_reference_volume.q11;
q12 = quadrature_responses_reference_volume.q12;
q13 = quadrature_responses_reference_volume.q13;

[sy_v sx_v sz_v] = size(q11);
[sy_o sx_o sz_o] = size(vol1);

F1 = quadrature_filters.F1;
F2 = quadrature_filters.F2;
F3 = quadrature_filters.F3;

p = [0 0 0 0 0 0 0 0 0 0 0 0]';

vol2prim = vol2;

dphi = zeros(sx_v*sy_v*sz_v,3);    
    
c = zeros(sx_v*sy_v*sz_v,3);
    
grad_x_dphi_n1 = zeros(sy_v,sx_v,sz_v);
grad_y_dphi_n2 = zeros(sy_v,sx_v,sz_v);
grad_z_dphi_n3 = zeros(sy_v,sx_v,sz_v);        

phase_gradients = zeros(sx_v*sy_v*sz_v,3,3);

x_motion_vectors = zeros(sy_o,sx_o,sz_o);
y_motion_vectors = zeros(sy_o,sx_o,sz_o);
z_motion_vectors = zeros(sy_o,sx_o,sz_o);

%for its = 1:max_its
p_diff = 1000;
p_old = p;
its = 0;
%while p_diff > 0.0001
for it = 1:max_its   
    
    %its = its + 1
    
    % FFT Convolution
    
    VOL2PRIM = fftn(vol2prim);

    q21_fft = fftshift(ifftn( VOL2PRIM .* F1));
    q22_fft = fftshift(ifftn( VOL2PRIM .* F2));
    q23_fft = fftshift(ifftn( VOL2PRIM .* F3));
    
    % Remove unnecesary data
    q21 = q21_fft( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
    q22 = q22_fft( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
    q23 = q23_fft( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
    
    [sy sx sz] = size(q21);
    
    dphi(:,1) = angle(q11(:).*conj(q21(:)));
    dphi(:,2) = angle(q12(:).*conj(q22(:)));
    dphi(:,3) = angle(q13(:).*conj(q23(:)));
        
    c(:,1) = abs(q11(:).*q21(:)).*((cos(dphi(:,1)/2)).^2);
    c(:,2) = abs(q12(:).*q22(:)).*((cos(dphi(:,2)/2)).^2);
    c(:,3) = abs(q13(:).*q23(:)).*((cos(dphi(:,3)/2)).^2);
    
     
    grad_x_dphi_n1(:,2:end-1,:) = angle(q11(:,3:end,:).*conj(q11(:,2:end-1,:)) + q11(:,2:end-1,:).*conj(q11(:,1:end-2,:)) + ...
                                        q21(:,3:end,:).*conj(q21(:,2:end-1,:)) + q21(:,2:end-1,:).*conj(q21(:,1:end-2,:)));
    
    
    grad_y_dphi_n2(2:end-1,:,:) = angle(q12(3:end,:,:).*conj(q12(2:end-1,:,:)) + q12(2:end-1,:,:).*conj(q12(1:end-2,:,:)) + ...
                                        q22(3:end,:,:).*conj(q22(2:end-1,:,:)) + q22(2:end-1,:,:).*conj(q22(1:end-2,:,:)));
    
    
    grad_z_dphi_n3(:,:,2:end-1) = angle(q13(:,:,3:end).*conj(q13(:,:,2:end-1)) + q13(:,:,2:end-1).*conj(q13(:,:,1:end-2)) + ...
                                        q23(:,:,3:end).*conj(q23(:,:,2:end-1)) + q23(:,:,2:end-1).*conj(q23(:,:,1:end-2)));                                                                             
                                    
    % Save all the phase gradients nicely...
    phase_gradients(:,1,1) = grad_z_dphi_n3(:);    
    phase_gradients(:,2,2) = grad_x_dphi_n1(:);
    phase_gradients(:,3,3) = grad_y_dphi_n2(:);
         
    %[certainties_1, certainties_2, certainties_3, phase_differences_1, phase_differences_2, phase_differences_3, phase_gradients_1, phase_gradients_2, phase_gradients_3] = calculate_certainties_phase_differences_and_phase_gradients(q11, q12, q13, q21, q22, q23);

    % Calculate A matrix and h vector
    certainties_1 = c(:,1);
    certainties_1 = reshape(certainties_1,sy,sx,sz);
     
    certainties_2 = c(:,2);
    certainties_2 = reshape(certainties_2,sy,sx,sz);
     
    certainties_3 = c(:,3);
    certainties_3 = reshape(certainties_3,sy,sx,sz);
     
    phase_differences_1 = dphi(:,1);
    phase_differences_1 = reshape(phase_differences_1,sy,sx,sz);
     
    phase_differences_2 = dphi(:,2);
    phase_differences_2 = reshape(phase_differences_2,sy,sx,sz);
     
    phase_differences_3 = dphi(:,3);
    phase_differences_3 = reshape(phase_differences_3,sy,sx,sz);
         
    phase_gradients_1 = grad_x_dphi_n1;
    phase_gradients_2 = grad_y_dphi_n2;
    phase_gradients_3 = grad_z_dphi_n3;                
    
    [A, h] = calculate_A_h_fast(certainties_1, certainties_2, certainties_3, phase_differences_1, phase_differences_2, phase_differences_3, phase_gradients_1, phase_gradients_2, phase_gradients_3);
       
    pp = inv(A)*h;

    % Update parameter vector
    p = p + pp;
    
    % Do a SVD and remove scaling
    p_matrix = reshape(p(4:end),3,3);
    p_matrix = p_matrix + diag([1 1 1]);
    [U,S,V] = svd(p_matrix);    
    p_matrix = U*V';
    p_matrix = p_matrix - diag([1 1 1]);    
    p(4:end) = reshape(p_matrix,3,3);
    p_diff = sum((p_old(:) - p(:)).^2);
    p_old = p;    
    
    
    
    % Find movement field
    x_motion_vectors(:) = p(1) + [x(:) y(:) z(:)]*p(4:6);
    y_motion_vectors(:) = p(2) + [x(:) y(:) z(:)]*p(7:9);
    z_motion_vectors(:) = p(3) + [x(:) y(:) z(:)]*p(10:12);
    
    vol2prim = interp3(x,y,z,vol2,x+x_motion_vectors,y+y_motion_vectors,z+z_motion_vectors,'linear');    % Generates NaN's    
    vol2prim(isnan(vol2prim)) = 0;
end

scalings(1) = S(1,1);
scalings(2) = S(2,2);
scalings(3) = S(3,3);

% Reshape parameter vector to transformation matrix
p_matrix = reshape(p(4:end),3,3);
% Add ones in the diagonal
p_matrix = p_matrix + diag([1 1 1]);
% Calculate rotation angles
angle1 = atan2(p_matrix(2,3),p_matrix(3,3))*180/pi;
c2 = sqrt(p_matrix(1,1)^2 + p_matrix(1,2)^2);
angle2 = atan2(-p_matrix(1,3),c2)*180/pi;
s1 = sind(angle1);
c1 = cosd(angle1);
angle3 = atan2(s1*p_matrix(3,1)-c1*p_matrix(2,1),c1*p_matrix(2,2)-s1*p_matrix(3,2))*180/pi;
rotations = [angle1, angle2, angle3];
    
compensated_volume = vol2prim;
registration_parameters = p;