clc; clear; close all;

% ============ CONFIGURATION — CHANGE THESE ============
base_path = 'D:\\ForCoding\\CodeProject\\Project\\MMIF\\Method-runs\\ref_repos\\pre_2019_env\\Harvard';          % <-- YOUR Harvard root folder
modality_pairs = {'CT-MRI', 'PET-MRI', 'SPECT-MRI'};
% ======================================================

output_base = fullfile('D:\\ForCoding\\CodeProject\\Project\\MMIF\\Method-runs\\ref_repos\\pre_2019_env\\Results\\SWTSWR');

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

    for i = 1:n_pairs
        % Load images
        y = imread(fullfile(folder1, files1(i).name));
        z = imread(fullfile(folder2, files2(i).name));
        image1 = double(y); if size(image1,3)==3, image1 = rgb2gray(image1); end
        image2 = double(z); if size(image2,3)==3, image2 = rgb2gray(image2); end

        if ~isequal(size(image1), size(image2)), error('Images must be same size'); end

        % ============ STEP 2: SWT Decomposition (Level 3) ============
        level = 3;
        [cA1, cH1, cV1, cD1] = swt2(image1, level, 'db1');
        [cA2, cH2, cV2, cD2] = swt2(image2, level, 'db1');

        % Fuse Approximation Coefficients
        SF1 = sqrt(mean2(diff(cA1(:,:,level),1,1).^2) + mean2(diff(cA1(:,:,level),1,2).^2));
        SF2 = sqrt(mean2(diff(cA2(:,:,level),1,1).^2) + mean2(diff(cA2(:,:,level),1,2).^2));
        total_SF = SF1 + SF2;
        if total_SF > 0
            w1 = SF1 / total_SF; w2 = SF2 / total_SF;
        else
            w1 = 0.5; w2 = 0.5;
        end
        fused_cA = cA1; 
        fused_cA(:,:,level) = w1 * cA1(:,:,level) + w2 * cA2(:,:,level);

        % Fuse High-Frequency Subbands
        window_size = 5; kernel = ones(window_size);
        fused_cH = zeros(size(cH1)); fused_cV = zeros(size(cV1)); fused_cD = zeros(size(cD1));
        for l = 1:level
            eH1 = sqrt(imfilter(cH1(:,:,l).^2, kernel, 'symmetric'));
            eH2 = sqrt(imfilter(cH2(:,:,l).^2, kernel, 'symmetric'));
            maskH = eH1 >= eH2; fused_cH(:,:,l) = maskH .* cH1(:,:,l) + (~maskH) .* cH2(:,:,l);

            eV1 = sqrt(imfilter(cV1(:,:,l).^2, kernel, 'symmetric'));
            eV2 = sqrt(imfilter(cV2(:,:,l).^2, kernel, 'symmetric'));
            maskV = eV1 >= eV2; fused_cV(:,:,l) = maskV .* cV1(:,:,l) + (~maskV) .* cV2(:,:,l);

            eD1 = sqrt(imfilter(cD1(:,:,l).^2, kernel, 'symmetric'));
            eD2 = sqrt(imfilter(cD2(:,:,l).^2, kernel, 'symmetric'));
            maskD = eD1 >= eD2; fused_cD(:,:,l) = maskD .* cD1(:,:,l) + (~maskD) .* cD2(:,:,l);
        end

        % Reconstruct fused SWT image
        fused_image_swt = iswt2(fused_cA, fused_cH, fused_cV, fused_cD, 'db1');

        % ============ STEP 3: WLS Post-Processing ============
        lambda_wls = 1.2;
        fused_image_wls = imguidedfilter(fused_image_swt, 'NeighborhoodSize', [9 9], ...
                                        'DegreeOfSmoothing', lambda_wls);

        % ============ STEP 4: Saliency-Based Final Fusion ============
        sal_swt = imgradient(fused_image_swt, 'sobel');
        sal_wls = imgradient(fused_image_wls, 'sobel');
        total_sal = sal_swt + sal_wls + eps;
        w_swt = sal_swt ./ total_sal;
        w_wls = sal_wls ./ total_sal;
        fused_final = w_swt .* fused_image_swt + w_wls .* fused_image_wls;


        % Local helpers
        calc_entropy = @(x) -sum(histcounts(x,256,'Normalization','probability') .* ...
                                log2(histcounts(x,256,'Normalization','probability') + eps));
        calc_scd = @(x, y, f) abs(corr2(x, f) - corr2(y, f));
        fmi = @(x, y) mutualinfo(double(edge(x, 'sobel')), double(edge(y, 'sobel')));

        % Save output
        [~, fname] = fileparts(files1(i).name);
        imwrite(uint8(255 * mat2gray(fused_final)), ...
                fullfile(out_dir, sprintf('%s.png', fname)));

        fprintf('  [%d/%d] done: %s\n', i, n_pairs, fname);
    end
end




