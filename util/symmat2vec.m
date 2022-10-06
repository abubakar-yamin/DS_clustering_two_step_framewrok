function [v] = symmat2vec(S,z)

%%% z define either to include diagnol or not. 0 to include, 1 to exclude

% S=tril(S);
% S(S==0) = [];
% x=x';

if nargin<2 || z==0
%     z=0;
    v_diag=diag(S);
    z=1;
end


[d,~,N] = size(S);
v1=[];
%     row = d+1;
    for i = 1:d
        tmp = S(i,i+z:end);
        v1 = horzcat(v1,tmp);
    end
    
if nargin<2 || z==0
    v = horzcat(v_diag',v1);
else
    v=v1;
end

end