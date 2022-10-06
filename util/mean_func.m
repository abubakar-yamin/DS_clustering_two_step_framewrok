function [d]=mean_func(F,distance)



    switch distance

                case 'karcher'   
                   sz=size(F,3);
                   for i = 1:sz
                       F_CM{i} = F(:,:,i);
                   end
                   [d,iter,theta] = karcher(F_CM{i:sz});    %% New Cluster Centers (Karcher mean

                case 'logE'  
                   d = geod_mean(F);                      % New Cluster Centers

                case 'eucl'
    %                 sz=size(F(:,:,A),3);
                   d = mean(F,3);

                case 'frech'
    %             sz=size(F(:,:,A),3);
                   d = mean(F,3);
%                    d = geod_mean(F); 
%                 sz=size(F,3);
%                 w = 1/sz;
%                 for i = 1:sz
%                     d = w*(d^0.5*F(:,:,i)*d^0.5)^0.5;
%                 end

                
    end

end
