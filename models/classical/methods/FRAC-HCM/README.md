# FRAC-HCM
This package contains the MATLAB code for *Multimodal medical image fusion based on edge detection and salient feature extraction. (BSPC 2025)*

Yimei Zheng, Xiaosong Li, Xiaoqi Cheng, Haishu Tan

-*[[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809425007608)]*

Usage of this code is free for research purposes only. 

Please refer to the above publication if you use this code. Thank you.

# Update
· [2025-7]  README.md is updated.

· [2025-6]  Codes and config files are public available.

# Citation
``` 
@article{zheng2025multimodal, 
  title = {Multimodal medical image fusion based on edge detection and salient feature extraction}, 
  author = {Zheng, Yimei and Li, Xiaosong and Cheng, Xiaoqi and Tan, Haishu}, 
  journal = {Biomedical Signal Processing and Control}, 
  volume = {110}, 
  pages = {108249}, 
  year = {2025}, 
  publisher = {Elsevier}
}
```

# Usage
This code is designed to run in the **MATLAB environment**.

**·** For Function/Anatomy image fusion (e.g., **PET/MRI**, **SPECT/MRI**, or **SPECT/GAD** fusion), please run:

      main_PET_MRI_or_SPECT_MRI.m

**·** For **CT/MRI** or **GAD/MRI** fusion, please run:

      main_CT_MRI_or_GAD_MRI.m

**·** For **MR-T1/MR-T2** fusion, please run:

      main_T1_T2.m

After opening the corresponding script in MATLAB, input the paths for the source images (**ImgA** and **ImgB**), then click Run to obtain the fusion result.

# Abstract
Multimodal medical image fusion integrates the complementary details from various imaging modalities for biomedical diagnostic applications. Although image fusion has been developing significantly in recent years, simultaneously retaining the edge and salient regions remains challenging. In this paper, we introduce a fusion approach based on edge detection and salient feature extraction. Particularly, a detection model that combines a fuzzy region-based active contour (FRAC) and a heat conduction matrix is proposed to extract the edge details. The edge details are fused based on the gradient magnitude. In addition, a local phase coherence intensity and pseudo-level set-based rule are developed to fuse the salient regions extracted by FRAC. The final fused image is reconstructed by integrating the preferred energy, edges, and salient features. Comparative experiments indicate that the proposed approach surpasses advanced methods in both quantitative and qualitative assessments. Our method achieved average improvements of 23.74 %, 0.95 %, 14.26 %, 12.39 %, 10.91 %, 21.06 %, 17.76 %, and 19.34 % for the QMI, QNCIE, SF, SD, AG, MS-SSIM, QS, and QAB/F metrics, respectively. The source code of this work is available at https://github.com/YimeiZheng/FRAC-HCM.

# 🙌 MCAFusion
**Illustration of our MCAFusion method.**

![FRAC-HCM 1](https://github.com/YimeiZheng/FRAC-HCM/blob/main/FRAC-HCM.png?raw=true)  

# Contact
Don't hesitate to contact me if you meet any problems when using this code.

Author: Yimei Zheng                                                         
Email: zhengyimei98@gmail.com.
