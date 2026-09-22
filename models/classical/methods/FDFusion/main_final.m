clc
close all 
clear all

% Get folder where main.m is located
repo_root = fileparts(mfilename('fullpath'));

% Add repo and all subfolders to MATLAB path
addpath(genpath(repo_root));

% ============ CONFIGURATION — CHANGE THESE ============
base_path = '.\Harvard';          % <-- YOUR Harvard root folder
modality_pairs = {'PET-MRI', 'SPECT-MRI'};
% ======================================================

output_base = fullfile('.\Results\FDFusion');

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
        AA = imread(fullfile(folder2, files2(idx).name));
        BB = imread(fullfile(folder1, files1(idx).name));


        if size(AA,3)>1
            AA=rgb2gray(AA);        
        end
        if size(BB,3)>1
            BB=rgb2gray(BB);        
        end

        A=im2double(AA);
        B=im2double(BB);
        %% Fusion method
        Q=6;
        AL=BLF_LS5(A,Q);
        BL=BLF_LS5(B,Q);
        AH=A-AL;
        BH=B-BL;
        %------------------------------------- High frequence------------------------------------------
        %% Significant edge
        I1=im2double(AH); 
        % 
        [gradientX, gradientY] = gradient(I1);
        gradientX(gradientX > 1) = 1;
        gradientX(gradientX < -1) = -1;
        gradientY(gradientY > 1) = 1;
        gradientY(gradientY < -1) = -1;
        Ix=gradientX;
        Iy=gradientY;

        edgeFIS = mamfis('Name', 'edgeDetection');
        %input
        edgeFIS = addInput(edgeFIS, [-1 1], 'Name', 'Ix');
        edgeFIS = addInput(edgeFIS, [-1 1], 'Name', 'Iy');

        sx=0.1;
        sy=0.1;
        edgeFIS = addMF(edgeFIS, 'Ix', 'gaussmf', [sx 0], 'Name', 'zero');
        edgeFIS = addMF(edgeFIS, 'Iy', 'gaussmf', [sy 0], 'Name', 'zero');

        edgeFIS = addOutput(edgeFIS, [0 1], 'Name', 'Iout');

        wa = 0.1;
        wb = 1;
        wc = 1;
        ba = 0;
        bb = 0;
        bc = 0.7;

        edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', [wa wb wc], 'Name', 'white');
        edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', [ba bb bc], 'Name', 'black');

        ruleList = [
            "If Ix is zero and Iy is zero then Iout is black";
            "If Ix is not zero or Iy is not zero then Iout is white"
        ];
        edgeFIS = addRule(edgeFIS, ruleList);


        Ieval1 = zeros(size(I1));
        for ii = 1:size(I1,1)
            Ieval1(ii,:) = evalfis([(Ix(ii,:));(Iy(ii,:));]',edgeFIS);
        end

        I2=im2double(BH);
        [gradientX, gradientY] = gradient(I2);
        gradientX(gradientX > 1) = 1;
        gradientX(gradientX < -1) = -1;
        gradientY(gradientY > 1) = 1;
        gradientY(gradientY < -1) = -1;
        Ix=gradientX;
        Iy=gradientY;

        edgeFIS2 = mamfis('Name', 'edgeDetection2');
        edgeFIS2 = addInput(edgeFIS2, [-1 1], 'Name', 'Ix');
        edgeFIS2 = addInput(edgeFIS2, [-1 1], 'Name', 'Iy');

        sx=0.1;
        sy=0.1;

        edgeFIS2 = addMF(edgeFIS2, 'Ix', 'gaussmf', [sx 0], 'Name', 'zero');
        edgeFIS2 = addMF(edgeFIS2, 'Iy', 'gaussmf', [sy 0], 'Name', 'zero');
        edgeFIS2 = addOutput(edgeFIS2, [0 1], 'Name', 'Iout');

        wa = 0.1;
        wb = 1;
        wc = 1;
        ba = 0;
        bb = 0;
        bc = 0.7;

        edgeFIS2 = addMF(edgeFIS2, 'Iout', 'trimf', [wa wb wc], 'Name', 'white');
        edgeFIS2 = addMF(edgeFIS2, 'Iout', 'trimf', [ba bb bc], 'Name', 'black');


        r1 = 'If Ix is zero and Iy is zero then Iout is black';
        r2 = 'If Ix is not zero or Iy is not zero then Iout is white';
        r = char(r1,r2);

        edgeFIS = addRule(edgeFIS, ruleList);


        Ieval2 = zeros(size(I2));
        for ii = 1:size(I2,1)
            Ieval2(ii,:) = evalfis([(Ix(ii,:));(Iy(ii,:));]',edgeFIS);
        end
        %%
        thr1 = graythresh(Ieval1);
        Ieval11 = imbinarize(Ieval1,thr1);
        thr2 = graythresh(Ieval2);
        Ieval22 = imbinarize(Ieval2,thr2);
        Ieval11=im2double(Ieval11);
        Ieval22=im2double(Ieval22);

        r2=4;
        eps = 0.02^2;
        S=4;
        Ieval111 =guidedfilter(AH, Ieval11, r2, eps);
        Ieval222 =guidedfilter(BH, Ieval22, r2, eps);
        FH1=Ieval111.*AH+Ieval222.*BH;

        T1=Ieval111.*AH;
        T2=Ieval222.*BH;
        %% insignificant edge
        AH2=AH-Ieval111.*AH;
        BH2=BH-Ieval222.*BH;
        %% --------------------------------------------------------------------
        FH=select_high_frequency(AH2, BH2);
        %figure,imshow(FH,[]);
        %% %-------------------------------------low frequency-------------------------------------
        map2=(AL>BL);
        FL=AL.*map2+BL.*(1-map2);
        %%
        FHH=FH1+FH;
        %% %------------------------------------fused image----------------------------------------
        F=FL+FHH;
        %% -------------------------------------reprocess------------------------------------------
        k0=0;
        n00=8;
        lambda = 0.045;
        if size(FHH, 3) == 3
            hIm = rgb2gray(FHH);
        else
            hIm = FHH;
        end

        if size(BH, 3) == 3
            lIm = rgb2gray(BH);
        else
            lIm = BH;
        end

        [Xhl,Xhs ] = inexact_alm_rpca(hIm, lambda);
        [Xll,Xls ] = inexact_alm_rpca(lIm, lambda);

        [PH]=Patches_get(hIm,k0,n00);
        [PL]=Patches_get(lIm,k0,n00);

        [P0]=Patches_get(Xhl,k0,n00);
        [P1]=Patches_get(Xhs,k0,n00);
        [P2]=Patches_get(Xll,k0,n00);
        [P3]=Patches_get(Xls,k0,n00);

            P00=P0;
            P11=P1;
            P22=P2;
            P33=P3;
        N=256;
        set1=double(P00);
        set2=double(P11);
        set3=double(P22);
        set4=double(P33);
        params1.data=(set1);
        params2.data=(set2);
        params3.data=(set3);
        params4.data=(set4);
        params1.Tdata =20;
        params1.dictsize = N;
        params1.iternum =150;
        params1.memusage = 'high';

        params2.Tdata =20;
        params2.dictsize = N;
        params2.iternum =150;
        params2.memusage = 'high';

        params3.Tdata =20;
        params3.dictsize = N;
        params3.iternum =150;
        params3.memusage = 'high';

        params4.Tdata =20;
        params4.dictsize = N;
        params4.iternum =150;
        params4.memusage = 'high';

        [Dksvd1] = ksvd(params1,'');
        [Dksvd2] = ksvd(params2,'');

        [Dksvd3] = ksvd(params3,'');
        [Dksvd4] = ksvd(params4,'');

        save('Dhl1_2.mat','Dksvd1');%
        save('Dhs1_2.mat','Dksvd2');%

        save('Dll1_2.mat','Dksvd3');%
        save('Dls1_2.mat','Dksvd4');%

        patch_size=4;  
        K=256;  

        %%   
        PHl=P0;
        PHs=P1;
        PLl=P2;
        PLs=P3;

        PHH=PH;
        PHll=PHl;
        PHss=PHs;
            
        PLL=PL;
        PLll=PLl;
        PLss=PLs;

        a=size(PHH);
        b=size(PLL);

        Dhl1=Dksvd1;
        Dhs1=Dksvd2;
        Dll1=Dksvd3;
        Dls1=Dksvd4;
        tic;
        [PP,Dhs,Dhl,Dls,Dll,Zls,Zll,H] = Du_Dicti_Train11(PHH ,PHss,PHll,Dhs1,Dhl1,PLL,PLss,PLll,Dls1,Dll1);
        toc;
        [ DD ] = UpadteD0( PP,Dhs,Dhl,Zls,Zll,H );
        save('D0_2.mat','DD');%  

        overlap = 6;                    
        epsilon=0.1;
        FHHH=sparse_fusion(FHH,BH,DD,overlap,epsilon);
        F2=FL+FHHH;
        % Save output
        [~, fname] = fileparts(files1(idx).name);
        imwrite(uint8(255 * mat2gray(F2)), ...
                fullfile(out_dir, sprintf('%s.png', fname)));

        fprintf('  [%d/%d] done: %s\n', idx, n_pairs, fname);
    end
end