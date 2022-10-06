function [d2] = Sym_distance_train_paral(x, distance_type)
%
% Compute the matrix of square distances between the data matrices
% in x
% x has form [x1, ...,xN]
% each xi is positive semi-definite
%
% gamma = regularization paramater (example: gamma = 10^(-9))
%

% x = x.*(x>0);    x(find(isnan(x)))=0;   x(find(isinf(x)))=0;


[n,n,N] = size(x);

d2 = ones(N);
% d5 = ones(N);
id_mat=eye(N);

switch distance_type
   
    case 'logE'
        log_mat=zeros(n,n,N);
        parfor i=1:N
            xi = squeeze(x(:,:,i));
%             log_mat(:,:,i) = logm(xi);
                [u1,s1] = eig(xi);
                log_mat(:,:,i) = u1*diag(log(diag(s1)))*u1';
        end
        for i=1:N
%             xi = squeeze(x(:,:,i));
          M1=squeeze(log_mat(:,:,i));
          d3=ones(1,N);
            parfor j=i+1:N
%                 if abs(i-j)>30
%                 d3(j) = sqrt(sum(sum((M1-log_mat(:,:,j)).^2)))
                 d3(j) = norm((M1-log_mat(:,:,j)),'fro');
                
%                 else
%                     d2(i,j)=0;
%                     d2(j,i)=0;
%                 end
                    
            end
           d2(i,:) = d3.*d2(i,:);
           d2(:,i) = d2(i,:);

        end
        d2=d2.*(~id_mat);
  
        case 'frech'
        for i=1:N
            M1=x(:,:,i);
            d3=ones(1,N);
            parfor j=i+1:N
%                 xj = squeeze(x(:,:,j));
                    p1 = M1 + x(:,:,j);
                    p2 = 2 * ((M1 * x(:,:,j)) ^ 0.5);
                    d3(j) = trace(p1-p2);
            end
           d2(i,:) = d3.*d2(i,:);
           d2(:,i) = d2(i,:);
        end 
        d2=d2.*(~id_mat);
        
        case 'karcher'
              for i=1:N
                xi = squeeze(x(:,:,i));
                for j=i+1:N
%                     if abs(i-j)>30
                    xj = squeeze(x(:,:,j));
                    d2(i,j)=dist_k(xi,xj);
                    d2(j,i) = d2(i,j);
%                     else
%                     d2(i,j)=0;
%                     d2(j,i)=0;
%                     end
                 end
              end 
              
        case 'eucl'
              for i=1:N
%                 M1=x(:,:,i);
%                 M1=M1(:);
                M1 = symmat2vec(x(:,:,i));
                d3=ones(1,N);
%                 d4=ones(1,N);
                parfor j=i+1:N
%                     xj = x(:,:,j);
                    xj=  symmat2vec(x(:,:,j));
                    d3(j)=norm(M1-xj);
%                     tmp=corrcoef(M1,xj);
%                     d4(j)=tmp(1,2);
                end
               d2(i,:) = d3.*d2(i,:);
               d2(:,i) = d2(i,:);
%                d5(i,:) = d4.*d5(i,:);
%                d5(:,i) = d5(i,:);
              end 
              d2=d2.*(~id_mat);
   
end

% for i=1:N
%     for j=1:N
%         if  d2(i,j)==0;
%                 d2(i,j)=max(max(d2));
%         end
%     end
% end


end





% % 
% %   p1 = a+ b;
% %                     p2 = 2 * ((a * b) ^ 0.5);
% %                     d22 = trace(p1-p2);



