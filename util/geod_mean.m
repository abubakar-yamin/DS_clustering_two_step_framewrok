function [mean_mt] = geod_mean(A)
%
%
% Compute the mean based on the Log-Euclidean distances
% between given array of symmetric, positive definite matrices


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [u1,s1] = eig(A);
% 
% [u2,s2] = eig(B);
% 
% logA = u1*diag(log(diag(s1)))*u1';
% 
% logB = u2*diag(log(diag(s2)))*u2';
% 
% si = 0.5 *  (logA + logB) ;
% 
% [u,s] = eig(si);
% 
% mean_mt = u*diag(exp(diag(s)))*u';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% d_mat=0.1*eye(size(A,1));

log_sum=0;

[N]=size(A,3);
    
for i=1:N
   mat=A(:,:,i);
%    mat=mat+d_mat;

   [u1,s1] = eig(mat);   
   logA = real(u1*diag(log(diag(s1)))*u1');

%    logA = real(logm(mat));
   
   log_sum = log_sum + logA;
end

si = (1/N) *  (log_sum) ;

% [u,s] = eig(si);
% mean_mt = u*diag(exp(diag(s)))*u';

mean_mt = real(expm(si));

end



% load('cov_mat_cerebrum.mat')
% a=cov_mat_cerebrum(:,:,1);
% b=cov_mat_cerebrum(:,:,2);
% gd_m=geod_mean(a,b);
% ed_m=(a+b)/2;
% [X,iter,theta]=karcher(c)
