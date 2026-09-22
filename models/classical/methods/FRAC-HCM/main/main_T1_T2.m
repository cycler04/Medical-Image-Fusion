%% Code for MR-T1/MR-T2 fusion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear all;close all;
addpath function

ImgA = imread('E:\sourceimage\T1+T2\T1\1.jpg');%image path of MR-T1
ImgB = imread('E:\sourceimage\T1+T2\T2\1.jpg');%image path of MR-T2
if size(ImgA,3)>1
    A=im2double(ImgA);
    A_YUV=ConvertRGBtoYUV(A);
    A=A_YUV(:,:,1);   
else
    A=im2double(ImgA);
end
if size(ImgB,3)>1
       B=im2double(ImgB);
    B=rgb2gray(B);
else
    B=im2double(ImgB);
end
G = fspecial('average',10);
[M,N,~] = size(A);
tic
%% Detail layers
E1=imfilter(A,G);
E2=imfilter(B,G);
A_H=A-E1; 
B_H=B-E2;  
%% edge detail layers
p=60;
T=3;
uA = FRAGL(A,M,N);
uB = FRAGL(B,M,N); 
segA = double(1<uA);
segB = double(1<uB);
mapsegA=double(uA>0.5);
mapsegB=double(uA>0.5);
sA=A.*mapsegA;
sB=B.*mapsegB;
edgeMap_imA=edgedetection(sA,M,N,p);
edgeMap_imB=edgedetection(sB,M,N,p);
edgeMap_imA=majority_consist_new(edgeMap_imA,T);   
edgeMap_imB=majority_consist_new(edgeMap_imB,T); 
edgeA=double((edgeMap_imA).*A_H); 
edgeB=double((edgeMap_imB).*B_H);
%% fusion of edge detail layers
[Gx1,Gy1] = imgradientxy(edgeA, 'sobel');     
[Gx2,Gy2] = imgradientxy(edgeB, 'sobel');
gradient_magnitude1 = sqrt(Gx1.^2 + Gy1.^2);   
gradient_magnitude2 = sqrt(Gx2.^2 + Gy2.^2);
gradient_EA=gradient_magnitude1.*edgeA;
gradient_EB=gradient_magnitude2.*edgeB;
MAPe=(gradient_EA>=gradient_EB);
EdgeFu=(~MAPe.*edgeB+MAPe.*edgeA); 
%% salient detail layers
sA1=A_H.*(EdgeFu==0); 
sB1=B_H.*(EdgeFu==0);
SA=sA1.*segA;
SB=sB1.*segB;
%% fusion of salient detail layers
[~, min_gr_ir, max_gr_ir, max_min_ir]  = LCP(SA,3); 
[~, min_gr_rgb, max_gr_rgb, max_min_rgb] = LCP(SB,3);  
DM=max(max_gr_ir,max_gr_rgb); 
DN=min(min_gr_ir,min_gr_rgb);
w1=(max_gr_ir)./(DM-DN);
w2=max_gr_rgb./(DM-DN); 
ww1=(w1.*uA).*((w1.*uA)>0);
ww2=(w2.*uB).*((w2.*uB)>0); 
mapSeg=ww1>ww2;
FusSeg=SA.*mapSeg+SB.* ~mapSeg; 
%% fusion of basic layers
inSA=sA1.*(FusSeg==0);
inSB=sB1.*(FusSeg==0);
map2=abs(E1+inSA)>abs(E2+inSB);
FE=(E1+inSA).*map2+~map2.*(E2+inSB);    
%% final fusion
F=EdgeFu+FusSeg+FE;
F_YUV=zeros(M,N,3);
F_YUV(:,:,1)=F;
F_YUV(:,:,2)=A_YUV(:,:,2);
F_YUV(:,:,3)=A_YUV(:,:,3);
final_F=(ConvertYUVtoRGB(F_YUV)); 
toc  


 