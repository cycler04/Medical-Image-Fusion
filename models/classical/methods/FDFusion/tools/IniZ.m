function [ A ] = IniZ( Data,DictMat,t)
% Xl=Patches_get(Xl,overlap,patch_size);
% Xs=Patches_get(Xs,overlap,patch_size);
% [ m,n ]   =   size(Xl);
% [ m1,n1 ]   =   size(Dl);
% x0=zeros(n1,n);
% opt.lambda = 0.01;  % param for regularizing param
% [Zl] = opt_fista_lasso(Dl,Xl,x0,opt,0);
% [Zs] = opt_fista_lasso(Ds,Xs,x0,opt,0);
% Gl=Dl'*Dl;
% Zl = omp2(Dl,Xl,Gl,epsilon);
% 
% Gs=Ds'*Ds;
% Zs = omp2(Ds,Xs,Gs,epsilon);
hRt1 = @(x) D1'*x;
tolA = 1e-3;
[A,x_debias,objective,times,debias_start,mses,max_svd] =TwIST(Data,DictMat,t, ...
         'Lambda', 2.2, ...
       'Debias',0,...
         'AT', hRt1, ... 
         'Monotone',1,...
       'Sparse', 1,...
         'Initialization',0,...
         'StopCriterion',1,...
       	 'ToleranceA',tolA,...
        'Verbose', 1);


end

