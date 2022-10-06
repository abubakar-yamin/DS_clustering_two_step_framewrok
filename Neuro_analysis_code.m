clear all;
close all;
clc;

load('HC_SCZ_data.mat') %%load the FC data matrices

distance = 'logE'; %%select the type of distance to be used
% % regularzie=1;  %% put 0 if dont want to regularize the matrices, 1 for  regularization
gamma = 10^-5;  %% Regularization value

C1 = HC;  %%% group 1 matrices
C2 = MS;  %%% group 2 matrices

%% combining the group matrices and creating labels
    allCM  = [];
    allCM  = cat(3,allCM,C1,C2);
    C1_lab = zeros(size(C1,3),1);
    C2_lab  = ones(size(C2,3),1);
    lab    = vertcat(C1_lab,C2_lab);
    
    
%% to visualize data and computing the distance matrix
[d2] = Sym_distance_train_paral(allCM, distance);
figure(); imshow(d2,[],'InitialMagnification','fit'), colorbar;colormap jet;
title('Similarity between subjects ');
xlabel('Subject Number');
ylabel('Subject Number');
axis on;
% % % xticks(1:1:40);
% % % yticks(1:1:40);

%% clustering and feature extraction
     
%%% applying clustering and computing distance of each subject from each cluster
        fprintf('\n DS on Whole Data \n');
        [LAB] = DS_clust_2(d2);
    
        [CENTS, DAL_tr, LAB_tr,K] = dist_2_cent(allCM,LAB, distance);
        k1=max(K);
%% Classification

    feat_set=[];
    feat_set=horzcat(feat_set,DAL_tr(:,1:k1));


K1=5; %%%number of folds 
for z=1:20    
    CVO{z} = cvpartition(lab,'KFold',K1);

    for ii=1:CVO{z}.NumTestSets  %% in each fold
    %     fprintf('\n Processing Fold %d of iteration %d\n',ii,z);

        tr = CVO{z}.training(ii);
        te = CVO{z}.test(ii);

        tr_lab{z}{:,ii} = lab(tr);
        te_lab{z}{:,ii} = lab(te);

        tr_dt = feat_set(tr,:);
        te_dt = feat_set(te,:);

        SVMMdl{z,ii}     = fitcsvm(tr_dt,tr_lab{z}{:,ii},'KernelScale','auto','Standardize',true,'OutlierFraction',0.0001,'prior','uniform');
        pred_lab       = predict(SVMMdl{z,ii},te_dt);
        EVAL_te{z}(ii,:)  = Evaluate_classifier(te_lab{z}{:,ii} ,pred_lab);
        Con{z,ii}        = confusionmat(te_lab{z}{:,ii} ,pred_lab);
        svm_weights{z}(:,ii) = SVMMdl{z,ii} .Beta;



        Mdl1{z,ii} = fitclinear(tr_dt,tr_lab{z}{:,ii},'Learner','logistic','Regularization','lasso','Lambda',0.025);
        pred_lab1 = predict(Mdl1{z,ii},te_dt);
        EVAL_te1{z}(ii,:) = Evaluate_classifier(te_lab{z}{:,ii} ,pred_lab1);
        Con1{z,ii}        = confusionmat(te_lab{z}{:,ii} ,pred_lab1);
        LASSO_weights{z}(:,ii) = Mdl1{z,ii} .Beta;
    
    end


  accu(:,z)  = mean(EVAL_te{z}(:,1));
  accu1(:,z) = mean(EVAL_te1{z}(:,1));
  
  %%% computing mean weights of each iteration of 5 fold CV
  svm_weig(z,:)    = mean(svm_weights{z},2);
  LASSO_weig(z,:)  = mean(LASSO_weights{z},2);


end

fprintf('\nMean accuracy for SVM \n');
mean(accu)*100

fprintf('\nMean accuracy for LASSO \n');
mean(accu1)*100


c1_c2_diff=k1(1);
for i=2:size(k1,2)
c1_c2_diff(i) = c1_c2_diff(i-1) + k1(i);
end


 %% code for thresholding of weights starts here
 %%% analyzing the classifier weights
    mean_LASSO_weig_abs = abs(mean(LASSO_weig));
    mean_SVM_weig_abs   = abs(mean(svm_weig));
    
    mean_LASSO_weig = (mean(LASSO_weig));
    mean_SVM_weig   = (mean(svm_weig));
  
%     figure();
%     plot(mean_SVM_weig,'r');
%     hold on;
%     plot(mean_LASSO_weig,'b');
% %     legend ('SVM','LASSO');
%     title('Average of weights of all iterations');
%     for i=1:size(k1,2)
%        line([c1_c2_diff(i) c1_c2_diff(i)],[0 2],'LineWidth',1)
%     end

%     figure();
%     plot(svm_weig','r');
%     hold on;
%     plot(LASSO_weig','b');
% %     legend ('SVM','LASSO');
%     title('All weights of all iterations');
    for i=1:size(k1,2)
       line([c1_c2_diff(i) c1_c2_diff(i)],[0 2],'LineWidth',1)
    end
    
    
    no_of_weig  = k1; %%no of weights to be analyzed = number of clusters
%     no_weig_sel = 6; %%no of weights to use for visualization of networks

    
    CENTS_f = CENTS;
    feat_set_f = feat_set;
    
    [val_f,idx_f] = maxk(mean_LASSO_weig_abs,no_of_weig); %%usign abs value to get both +ve and -ve peak values
    
    
%% showing distribution of subjects around each cluster    
    figure();
    j=1;
    for i=1:size(idx_f,2)    
        
%         [subnet_no_f(i),clust_no_f(i)] = find(sn_all==idx_f(i));
        cent_f{i} = CENTS_f(:,:,idx_f(i));
        bns1 = min(feat_set_f(:,idx_f(i))):0.1:max(feat_set_f(:,idx_f(i)));
        cnts1 = histcounts(feat_set_f(1:sum(lab==0),idx_f(i)),bns1);
        cnts2 = histcounts(feat_set_f(sum(lab==0)+1:end,idx_f(i)),bns1);
        
        subplot(no_of_weig,2,j);
        bar(bns1(1:end-1)+(bns1(2)-bns1(1))/2,[cnts1;cnts2]')
        title(['Distribution of subjects for ',num2str(i),' highest weight cluster']);
        j=j+1;
        subplot(no_of_weig,2,j);
        imshow(cent_f{i},[ ],'InitialMagnification','fit'), colorbar,colormap('jet');
        title(['Centroid of ',num2str(i),' - ',num2str(idx_f(i)),' highest weight cluster']);
        j=j+1;        
    end
    
    dist_feat_mat = [];
    
%% Permutation test on classifier weights to select significant features
no_iterat = 100;
K1=5; %%% number of folds
 
for z=1:no_iterat   
    
    dist_feat_mat_shuff = Shuffle(feat_set_f , 2); %%shuffling the feature values

    CVO{z} = cvpartition(lab,'KFold',K1);

    for ii=1:CVO{z}.NumTestSets
%     fprintf('\n Processing Fold %d of iteration %d\n',ii,z);
    
    tr = CVO{z}.training(ii);
    te = CVO{z}.test(ii);
    
    tr_lab{z}{:,ii} = lab(tr);
    te_lab{z}{:,ii} = lab(te);
    
    tr_dist_feat_mat = dist_feat_mat_shuff(tr,:);
    te_dist_feat_mat = dist_feat_mat_shuff(te,:);
    
    
    SVMMdl{z,ii}     = fitcsvm(tr_dist_feat_mat,tr_lab{z}{:,ii},'KernelScale','auto','Standardize',true,'OutlierFraction',0.0001,'prior','uniform');
    pred_lab       = predict(SVMMdl{z,ii},te_dist_feat_mat);
    EVAL_te{z}(ii,:)  = Evaluate_classifier(te_lab{z}{:,ii} ,pred_lab);
    Con{z,ii}        = confusionmat(te_lab{z}{:,ii} ,pred_lab);
    svm_weights{z}(:,ii) = SVMMdl{z,ii}.Beta;
%    zz=horzcat(zz,svm_weights{z}(:,ii));
   

    Mdl1{z,ii} = fitclinear(tr_dist_feat_mat,tr_lab{z}{:,ii},'Learner','logistic','Regularization','lasso','lambda',0.02);
    pred_lab1 = predict(Mdl1{z,ii},te_dist_feat_mat);
    EVAL_te1{z}(ii,:) = Evaluate_classifier(te_lab{z}{:,ii} ,pred_lab1);
    Con1{z,ii}        = confusionmat(te_lab{z}{:,ii} ,pred_lab1);
    LASSO_weights{z}(:,ii) = Mdl1{z,ii} .Beta;

    end

  accu(:,z)  = mean(EVAL_te{z}(:,1));
  accu1(:,z) = mean(EVAL_te1{z}(:,1));
  
    LASSO_weig_mean(:,z) = abs(mean(LASSO_weights{z},2));
    SVM_weig_mean(:,z)   = abs(mean(svm_weights{z},2));
    
    LASSO_weig_mean_permu(:,z) = (mean(LASSO_weights{z},2));
    SVM_weig_mean_permu(:,z)   = (mean(svm_weights{z},2));
    
end
    
%     figure();
%     plot(SVM_weig_mean_permu,'r');
%     hold on;
%     plot(LASSO_weig_mean_permu,'b');
%     legend ('SVM','LASSO');
%     title('Average of weights for random permutation run');
% 
%      
% [cnts11,bns11]=histcounts(SVM_weig_mean_permu,'BinWidth',0.03);
% figure();bar(bns11(1:end-1)+(bns11(2)-bns11(1))/2,[cnts11])
% title('Hist of SVM weights');
% [cnts12,bns12]=histcounts(LASSO_weig_mean_permu,'BinWidth',0.03);
% figure();bar(bns12(1:end-1)+(bns12(2)-bns12(1))/2,[cnts12])
% title('Hist of LASSO weights');
    
%% selecting the significant centroid/future

mx_vl = max(max(SVM_weig_mean_permu));
up_vl=0;
while up_vl <= 0.05
    up_vl = sum(sum(SVM_weig_mean_permu>mx_vl))/sum(sum(SVM_weig_mean_permu<100000));
    mx_svm_thr_vl = mx_vl;
    mx_vl = mx_vl - 0.01;
end

mi_vl = min(min(SVM_weig_mean_permu));
lw_vl=0;
while lw_vl <= 0.05
    lw_vl = sum(sum(SVM_weig_mean_permu<mi_vl))/sum(sum(SVM_weig_mean_permu<100000));
    mi_svm_thr_vl = mi_vl;
    mi_vl = mi_vl + 0.01;
end


mx_vl = max(max(LASSO_weig_mean_permu));
up_vl=0;
while up_vl <= 0.05
    up_vl = sum(sum(LASSO_weig_mean_permu>mx_vl))/sum(sum(LASSO_weig_mean_permu<100000));
    mx_lasso_thr_vl = mx_vl;
    mx_vl = mx_vl - 0.01;
end

mi_vl = min(min(LASSO_weig_mean_permu));
lw_vl=0;
while lw_vl <= 0.05
    lw_vl = sum(sum(LASSO_weig_mean_permu<mi_vl))/sum(sum(LASSO_weig_mean_permu<100000));
    mi_lasso_thr_vl = mi_vl;
    mi_vl = mi_vl + 0.01;
end

fprintf('\nSVM Upper limit %d: & and lower limit %d\n', mx_svm_thr_vl,mi_svm_thr_vl);
fprintf('\nLASSO Upper limit %d: & and lower limit %d\n', mx_lasso_thr_vl,mi_lasso_thr_vl);

 %%% plotting the mean of feature weights and their obtained threshold
    figure();
    h(1)=plot(mean_SVM_weig,'r','LineWidth',2);
    hold on;
    h(2)=plot(mean_LASSO_weig,'b','LineWidth',2);
% %     legend ('SVM','LASSO');

    h(3)=line([0,k1(end)],[mx_svm_thr_vl,mx_svm_thr_vl],'LineWidth',1,'Color','#D95319');
    h(4)=line([0,k1(end)],[mi_svm_thr_vl,mi_svm_thr_vl],'LineWidth',1,'Color','#D95319');
    
    h(5)=line([0,k1(end)],[mx_lasso_thr_vl,mx_lasso_thr_vl],'LineWidth',1,'Color','#77AC30');
    h(6)=line([0,k1(end)],[mi_lasso_thr_vl,mi_lasso_thr_vl],'LineWidth',1,'Color','#77AC30');

    title('Average of weights of initial full data run of HC vs RR');
    xlabel('Feature Number');
    ylabel('Classifier Weights');


[lasso_weig_val,lasso_weig_idx] = maxk(mean_LASSO_weig,no_of_weig);
[svm_weig_val,svm_weig_idx] = maxk(mean_SVM_weig,no_of_weig);

[row,col] = find(lasso_weig_val>mx_lasso_thr_vl);
% sel_feat_lasso(1) = lasso_weig_idx(row(1),col(1));

[row,col] = find(svm_weig_val>mx_svm_thr_vl);
% sel_feat_svm(1) = svm_weig_idx(row(1),col(1));

%% selecting the threshold to seperate the subjects around the cluster centroid

svm_acc11=[];
thr_val = [];
th_val = [];
prompt = 'Select the feature number for analysis? ';
feat_no = input(prompt);

%%% showing histogram of distribution of subjects around the centroid
bns1 = min(feat_set_f(:,feat_no)):0.1:max(feat_set_f(:,feat_no));
cnts1 = histcounts(feat_set_f(1:sum(lab==0),feat_no),bns1);
cnts2 = histcounts(feat_set_f(sum(lab==0)+1:end,feat_no),bns1);
figure();
bar(bns1(1:end-1)+(bns1(2)-bns1(1))/2,[cnts1;cnts2]')
%         hold on;
%         line([thresh(i),thresh(i)],[0,max(max(cnts1,cnts2))],'Color','red','LineStyle','--', 'LineWidth',3)
title(['Distribution of subjects for ',num2str(feat_no),' highest weight cluster']);


aa = feat_set(:,feat_no);
mx = max(aa);
mi = min(aa);
thr_val = mi+(range(aa)/6);

THs=linspace(min(aa),max(aa),100);
for currTh=1:length(THs)
    lblsEst=aa>THs(currTh);
    acc(currTh)=mean(lblsEst==lab);
end
figure;plot(THs,acc)
[~,maxPos]=max(acc);
th_val(:,1)=THs(maxPos);

THs=linspace(min(aa),max(aa),100);
for currTh=1:length(THs)
lblsEst=aa<THs(currTh);
acc(currTh)=mean(lblsEst==lab);
end
% figure;plot(THs,acc)
[~,maxPos]=max(acc);
th_val(:,2)=THs(maxPos);
th_val(:,3)=(th_val(:,1)+th_val(:,2))/2;

%%% select the suitable threshold value manualy. interpretation is required
%%% for selection

% %% grouping the subjects 

thr_sel = input(['Select the threshold value from 1)' num2str(th_val(:,1)) ', 2) '...
    num2str(th_val(:,2)) ' , 3)' num2str(th_val(:,3)) ', or enter the desired value?']);
if thr_sel == 1 || thr_sel == 2 || thr_sel == 3  
    thresh = th_val(:,thr_sel);
else
thresh=thr_sel;
end

    c1_sel=[];
    hc_sel1=[];
    c2_sel=[];
    p_sel1=[];
    
    %%% selecting subjects below the threshold (one class) and above threshold (other class) 
%     [subnet_no,clust_no] =find(sn_all==feat_no(ii));
    c1_sel1 = feat_set_f(1:sum(lab==0),feat_no)<(thresh);
    c2_sel1 = logical(vertcat(zeros(size(c1_sel,1),1),feat_set_f(sum(lab==0)+1:end,feat_no)>(thresh)));
    
%     sum(c1_sel)
%     sum(c2_sel)
    
%     if(sum(c1_sel)<=sum(c2_sel))
        c1_sel2 = feat_set_f(1:sum(lab==0),feat_no)>(thresh);
        c2_sel2 = logical(vertcat(zeros(size(c1_sel,1),1),feat_set_f(sum(lab==0)+1:end,feat_no)<(thresh)));
%     end
    
%     sum(c1_sel)
%     sum(c2_sel)

%%% select the groups on either sides of threshold
grp_sel = input(['Select the group side along the threshold 1)' num2str(sum(c1_sel1)) '&' num2str(sum(c2_sel1)) ' or , 2)' num2str(sum(c1_sel2)) '&' num2str(sum(c2_sel2)) '?']);

if grp_sel == 1
    c1_sel = c1_sel1;
    c2_sel = c2_sel1;
elseif grp_sel == 2
    c1_sel = c1_sel2;
    c2_sel = c2_sel2;
end

%selecting the subjects from groups and compute the mean (group reference connectome)
    [c1_mean] = mean_func(allCM(:,:,c1_sel),distance);
    [c2_mean] = mean_func(allCM(:,:,c2_sel),distance);
    
    sz = size(c1_mean);
    %%% computing the difference between group reference connectome
    c1_mean = c1_mean.*~eye(sz);
    c2_mean  = c2_mean.*~eye(sz);
    c1_c2_diff = (c1_mean-c2_mean);


% plot_img(c1_mean,c2_mean,c1_c2_diff)
% 
%% plotting the  the mean and differences
% function plot_img(c1_mean,c2_mean,c_mean_diff)
   

mx = max(max(c1_mean, [], 'all'), max(c2_mean, [], 'all'));
mi = min(min(c1_mean, [], 'all'), min(c2_mean, [], 'all'));

mx1 = max(c1_c2_diff, [], 'all');
mi1 = -mx1; %min(c1_c2_diff, [], 'all');


Cmap = [];
Cmap1 = summer(64);
Cmap2 = flipud(autumn(64));
Cmap = flipud(vertcat(Cmap1,Cmap2));
Cmap = Cmap (49:end,:);
% Cmap3 = customcolormap_preset('red-white-blue');
% Cmap3(32,:) = [1,1,1]; 
load('Cmap1');


    ax(1)=figure(); imshow(c1_mean,[mi mx],'InitialMagnification','fit'), colorbar;
    cH=colorbar;
        colormap(ax(1),Cmap),set(cH,'FontSize',12);
%     axis_lab(7)
        title('C1 Means ','FontSize',18);
%           a = get(gca,'XTickLabel');
%     set(gca,'XTickLabel',a,'FontName','Times','fontsize',15)
    
    ax(1)=figure(); imshow(c2_mean,[mi mx],'InitialMagnification','fit'), colorbar;
cH=colorbar;
    colormap(ax(1),Cmap),set(cH,'FontSize',12);
% axis_lab(7)
    title('C2 Means ','FontSize',18);
%       a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',15)


    ax(1)=figure(); imshow(c1_c2_diff,[mi1 mx1],'InitialMagnification','fit'), colorbar;
cH=colorbar;
    colormap(ax(1),Cmap),set(cH,'FontSize',12);
% axis_lab(7)
    title('Diff of C1-C2 Means ','FontSize',14);
%      a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',15)

%%% computing the the top 2.5% connection on both side (-ve and +ve side)
mx_vl = max(max(c1_c2_diff));
up_vl=0;
while up_vl <= 0.025
    up_vl = sum(sum(c1_c2_diff>mx_vl))/sum(sum(c1_c2_diff<100000));
    mx_vl_t1 = mx_vl;
    mx_vl = mx_vl - 0.01;
end

mi_vl = min(min(c1_c2_diff));
lw_vl=0;
while lw_vl <= 0.025
    lw_vl = sum(sum(c1_c2_diff<mi_vl))/sum(sum(c1_c2_diff<100000));
    mi_vl_t1 = mi_vl;
    mi_vl = mi_vl + 0.01;
end

a1 = c1_c2_diff>mx_vl;
a2 = c1_c2_diff<mi_vl;
a3 = or(a1,a2);
a23 = c1_c2_diff.*(a3);

mx1 = max(a23, [], 'all');
mi1 = -mx1; %min(c1_c2_diff, [], 'all');

ax(3)=figure(); imshow(a23,[-0.9 0.9],'InitialMagnification','fit'), colorbar;
cH=colorbar;
colormap(ax(3),Cmap1),set(cH,'FontSize',8);
title('significance connections only')
% axis_lab(1)
      th_val = get(gca,'XTickLabel');
set(gca,'XTickLabel',th_val,'FontName','calibri','fontsize',8)


%%% The End %%%