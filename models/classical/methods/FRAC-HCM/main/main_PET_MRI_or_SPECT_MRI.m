%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear all;close all;
addpath function
clc
close all 
clear all

% Get folder where main.m is located
repo_root = fileparts(mfilename('fullpath'));

% Add repo and all subfolders to MATLAB path
addpath(genpath(repo_root));

% ============ CONFIGURATION — CHANGE THESE ============
base_path = 'D:\\ForCoding\\CodeProject\\Project\\MMIF\\Method-runs\\ref_repos\\pre_2019_env\\Harvard';          % <-- YOUR Harvard root folder
modality_pairs = {'PET-MRI', "SPECT-MRI"};
% ======================================================

output_base = fullfile('D:\\ForCoding\\CodeProject\\Project\\MMIF\\Method-runs\\ref_repos\\pre_2019_env\\Results\\FRAC-HCM');

for m = 1:length(modality_pairs)
    pair = modality_pairs{m};
    pair_path = fullfile(base_path, pair);
    if ~isfolder(pair_path), continue; end

    % Auto-detect the two subfolders (CT & MRI, PET & MRI, etc.)
    sub = dir(pair_path);
    sub = sub([sub.isdir] & ~startsWith({sub.name}, '.'));
    names = {sub.name};
    mri_idx = find(strcmpi(names, 'MRI'));
    other_idx = find(~strcmpi(names, 'MRI'));

    folder1 = fullfile(pair_path, names{mri_idx});    % always MRI
    folder2 = fullfile(pair_path, names{other_idx});  % CT, PET, or SPECT

    % Collect image files
    exts = {'*.png','*.jpg','*.tif','*.bmp'};
    files1 = []; files2 = [];
    for e = 1:length(exts)
        files1 = [files1; dir(fullfile(folder1, exts{e}))];
        files2 = [files2; dir(fullfile(folder2, exts{e}))];
    end

    n_pairs = min(length(files1), length(files2));
    fprintf('\n>>> %s: %d pairs\n', pair, n_pairs);

    % Output folder
    out_dir = fullfile(output_base, pair);
    if ~isfolder(out_dir), mkdir(out_dir); end

    for idx = 1:n_pairs
        % Load images
        ImgA = imread(fullfile(folder2, files2(idx).name));
        ImgB = imread(fullfile(folder1, files1(idx).name));
        if size(ImgA,3)>1
            A = im2double(ImgA);
            A_YUV = ConvertRGBtoYUV(A);
            A = A_YUV(:,:,1);
            isColorA = true;
        else
            A = im2double(ImgA);
            A_YUV = [];       % no chrominance channels available
            isColorA = false;
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
        segA = double((0.7<uA)&(uA<1));
        segB = double((0.7<uB)&(uB<1));
        edgeMap_imA=edgedetectionA(A,uA,M,N,p);
        edgeMap_imB=edgedetectionB(B,uB,M,N,p);
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
        [~, min_gr_ir,  max_gr_ir,  max_min_ir]  = LCP(SA,3); 
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
        F = EdgeFu + FusSeg + FE;

        if isColorA
            F_YUV = zeros(M, N, 3);
            F_YUV(:,:,1) = F;
            F_YUV(:,:,2) = A_YUV(:,:,2);
            F_YUV(:,:,3) = A_YUV(:,:,3);
            final_F = ConvertYUVtoRGB(F_YUV);
        else
            % Both inputs are grayscale — output is grayscale
            final_F = F;
        end
        toc

        % Save output
        [~, fname] = fileparts(files1(idx).name);
        imwrite(uint8(255 * mat2gray(final_F)), ...
                fullfile(out_dir, sprintf('%s.png', fname)));
    end
end
        