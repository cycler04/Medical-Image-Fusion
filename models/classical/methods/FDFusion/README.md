# Multi-modality image fusion using fuzzy set theory and compensation dictionary learning
Authors: Yuchan Jie, Xiaosong Li, Tianshu Tan, Lemiao Yang, Mingyi Wang

Published in: Optics and Laser Technology
Y.C.Jie, X.S. Li, T.S Tan, L.M. Yang, M.Y. Wang,"Multi-modality image fusion using fuzzy set theory and compensation dictionary learning", Opt. Laser Technol. 181 (2025) 112001.
# Note
Our model is in the "main_final.m" file, run it to obtain the fused results.
# Abstract
Multi-modality image fusion aims to integrate complementary information from different modalities to produce superior images for advanced visual tasks. Existing fusion methods often struggle with effectively extracting edge and details, especially under complex conditions where images contain mixed background-target information or are degraded by subtle noise. This often results in incomplete or blurred representation of edge and other detail information. To address these challenges, we propose FDFusion, a novel fusion approach leveraging fuzzy set
theory and compensation dictionary learning. First, we introduce the SVB-LS filter, a novel tool to achieve image smoothing while preserving edges and global structures simultaneously. This filter plays a crucial role in achieving “structure-background” decomposition, which is essential for enhancing the extraction of cross-modal information during the fusion process. Also, to preserve significant edges from the source images, we propose a structure layer saliency fusion rule utilizing a fuzzy inference system. Additionally, for non-salient structures, we introduce an intuitionistic fuzzy set similarity measure designed to comprehensively capture the membership, non-membership, and hesitation data, which are critical for managing the complex textures in non-salient detail features. Furthermore, to counteract the loss of detail resulting from the decomposition and reconstruction phases of the fusion process, we develop a compensation dictionary learning strategy aimed at enhancing the visibility and clarity of the fusion results. Extensive experiments demonstrate that FDFusion outperforms the state-of-the-art methods in terms of fusion performance and exhibits superior adaptability to complex scenes. Additionally, it also excels in segmentation and detection tasks, showcasing its board applicative potential.
# The framework of fusion method
![总流程](https://github.com/user-attachments/assets/79b8c448-62d5-4698-be91-e5bfceb43f2f)
# Experimental results
![实验](https://github.com/user-attachments/assets/2450e720-ec01-452f-98b1-f2bf42a2ff73)

If you have any question about this code, feel free to reach me(jyc981214@163.com)
# Citation
Jie, Y., Li, X., Tan, T., Yang, L., and Wang, M.: ‘Multi-modality image fusion using fuzzy set theory and compensation dictionary learning’, Optics & Laser Technology, 2025, 181, pp. 112001.

```bibtex
@article{JIE2025112001,
title = {Multi-modality image fusion using fuzzy set theory and compensation dictionary learning},
journal = {Optics & Laser Technology},
volume = {181},
pages = {112001},
year = {2025},
issn = {0030-3992},
doi = {https://doi.org/10.1016/j.optlastec.2024.112001},
url = {https://www.sciencedirect.com/science/article/pii/S0030399224014592},
