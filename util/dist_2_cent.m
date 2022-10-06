function [CENTS, DAL, LAB, K] = dist_2_cent(F, labels, distance)
%% Information
%%% This algo is to compute the centroid of given labels and  distance of each matrix with given
%%% these centroid and return the distance and from each centroid and label for
%%% each matrix.
%%
K=unique(labels);
fprintf('\nTotal Number of Clusters are: %d\n',max(K));
idx=1;
 for i3 = 1:max(K)
      A = (labels == K(idx,:));                          % Cluster K Points
       if sum(A)==0
          fprintf('\n\n\n pause due to NaN value in CENTS\n\n\n');
          pause();pause();pause();
      end
      
      [CENTS(:,:,idx)]=mean_func(F(:,:,A),distance);
      
      if sum(sum(isnan(CENTS(:,:,idx))))>=1
          fprintf('\n\n\n pause due to NaN value in CENTS\n\n\n');
          pause();pause();pause();
%             CENTS(:,:,i3)=F(:,:,randi(size(F,3),1,1));   %%if NaN center, then replace it with random point in data
      end
idx=idx+1;
 end


% while n<=KMI
fprintf('\nComputing Disance to Centroid for K:%d\n',max(K));        
   for i2 = 1:size(F,3)
      for j = 1:max(K)  
%         DAL(i,j) = norm(F(:,:,1) - CENTS(j,:));     
        DAL(i2,j) = fdist_func(F(:,:,i2),CENTS(:,:,j),distance);
      end
      [Distance CN] = min(DAL(i2,1:max(K)));                % 1:K are Distance from Cluster Centers 1:K 
      DAL(i2,max(K)+1) = CN;                                % K+1 is Cluster Label
      LAB(i2,1) = CN;
      DAL(i2,max(K)+2) = Distance;                          % K+2 is Minimum Distance
   end
% %    fprintf('\nUpdating Centroid of iteration:%d for K:%d\n',n,K);  
% %    for i3 = 1:K
% %       A = (DAL(:,K+1) == i3);                          % Cluster K Points
% %        if sum(A)==0
% %           fprintf('\n\n\n pause due to NaN value in CENTS\n\n\n');
% %           pause();pause();pause();
% %       end
% %       
% %       [CENTS(:,:,i3)]=mean_func(F(:,:,A),distance);
% %       
% %       if sum(sum(isnan(CENTS(:,:,i3))))>=1
% %           fprintf('\n\n\n pause due to NaN value in CENTS\n\n\n');
% %           pause();pause();pause();
% % %             CENTS(:,:,i3)=F(:,:,randi(size(F,3),1,1));   %%if NaN center, then replace it with random point in data
% %       end
% % 
% %    end
% %    
% %    prev_DAL(:,n)=DAL(:,K+1);
% %    
% %    if(n>=3)
% %         fprintf('\nCovergence Check for iteration:%d for K:%d\n',n,K);
% %        tf = isequal(prev_DAL(:,n-2),prev_DAL(:,n-1),prev_DAL(:,n));
% %    else
% %     tf=0;
% %    end
% %    
% %    if (tf==1 || n==200)
% %        fprintf('\n****************   Covergence Achieved for K:%d at iteration:%d   **************\n',K,n);
% %        break
% %    else
% %        fprintf('\nCovergence Not Achieved, increment in iteration:\n');
% % %        KMI=KMI+1;
% %    end
% %    n=n+1;
% end
   
%% Plot   
%     clf
%     figure(1)
%     hold on
%      for i = 1:K
%     PT = F(DAL(:,K+1) == i,:);                            % Find points of each cluster    
%     plot(PT(:,1),PT(:,2),CV(2*i-1:2*i),'LineWidth',2);    % Plot points with determined color and shape
%     plot(CENTS(:,1),CENTS(:,2),'*k','LineWidth',7);       % Plot cluster centers
%      end
%     hold off
%     grid on
%     pause(0.1)

end