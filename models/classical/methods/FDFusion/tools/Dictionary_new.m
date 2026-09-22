function Dksvd=Dictionary_new(I1,I2)

addpath ksvdbox13;
addpath ompbox10;

k=6;    
n0=8;     
N=256;     

[row,line,~]=size(I1);
%% 
% I1_new1=I1(1:2:row,1:2:line,:);    
I1_new2=I1(1:3:row,1:3:line,:);     
I1_new3=I1(1:4:row,1:4:line,:);      
I1_new4=I1(1:5:row,1:5:line,:);      
I1_new5=I1(1:6:row,1:6:line,:);      
% I2_new1=I2(1:2:row,1:2:line,:);   
I2_new2=I2(1:3:row,1:3:line,:);      
I2_new3=I2(1:4:row,1:4:line,:);      
I2_new4=I2(1:5:row,1:5:line,:);      
I2_new5=I2(1:6:row,1:6:line,:);     

% [P_E1,P_C1]=Patchesget_new(I1_new1,I2_new1,k,n0);    
[P_E2,P_C2]=Patchesget_new(I1_new2,I2_new2,k,n0);
[P_E3,P_C3]=Patchesget_new(I1_new3,I2_new3,k,n0);
[P_E4,P_C4]=Patchesget_new(I1_new4,I2_new4,k,n0);
[P_E5,P_C5]=Patchesget_new(I1_new5,I2_new5,k,n0);

% EC1=[P_E1,P_C1];
EC2=[P_E2,P_C2];
EC3=[P_E3,P_C3];
EC4=[P_E4,P_C4];
EC5=[P_E5,P_C5];
% [P0]=[EC1 EC2 EC3 EC4];
[P0]=[ EC2 EC3 EC4 EC5];

set=P0;
set=double(set/255);
params.data=(set);
[r,z]=size(set);
%params.data=X;
%params.data=X;
params.Tdata =20;
params.dictsize = N;
params.iternum =50;       
params.memusage = 'high';
% tic
[Dksvd,g,err] = ksvd(params,'');  
% toc;