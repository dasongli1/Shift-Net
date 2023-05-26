# A Simple Baseline for Video Restoration with Spatial-temporal Shift (CVPR 2023)
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_A_Simple_Baseline_for_Video_Restoration_With_Grouped_Spatial-Temporal_Shift_CVPR_2023_paper.pdf)
**|** 
[supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Li_A_Simple_Baseline_CVPR_2023_supplemental.pdf)
**|** 
[arxiv](https://arxiv.org/abs/2206.10810)
**|** 
[project page](https://dasongli1.github.io/publication/grouped-shift-net/)

This repository is the official PyTorch implementation of "A Simple Baseline for Video Restoration with Spatial-temporal Shift"

> **Abstract:** *Video restoration, which aims to restore clear frames from degraded videos, has numerous important applications. The key to video restoration depends on utilizing inter-frame information. However, existing deep learning methods often rely on complicated network architectures, such as optical flow estimation, deformable convolution, and cross-frame self-attention layers, resulting in high computational costs. In this study, we propose a simple yet effective framework for video restoration. Our approach is based on grouped spatial-temporal shift, which is a lightweight and straightforward technique that can implicitly capture inter-frame correspondences for multi-frame aggregation. By introducing grouped spatial shift, we attain expansive effective receptive fields. Combined with basic 2D convolution, this simple framework can effectively aggregate inter-frame information. Extensive experiments demonstrate that our framework outperforms the previous state-of-the-art method, while using less than a quarter of its computational cost, on both video deblurring and video denoising tasks. These results indicate the potential for our approach to significantly reduce computational overhead while maintaining high-quality results.* 

### Pre-trained Models
| Task                                 | Dataset | Model      | Visual Results  |
| :----------------------------------- | :------ | :----------- | :------------------------ |
| Video Deblurring                     | GoPro   | Ours+ | [gdrive](https://drive.google.com/file/d/1f79zxmCL-ygVmoJd86OT6uksPGk2BpL0/view?usp=share_link) |
| Video Deblurring                     | GoPro   | Ours-s | [gdrive](https://drive.google.com/file/d/1WnFZRnXN9ZJebMZaAZF6a0c8f3CrLjR_/view?usp=share_link) |
| Video Deblurring                     | DVD   | Ours+ | [gdrive](https://drive.google.com/file/d/1vPQkAznRaVawQOMOuvhnCD8DKS2RQToM/view?usp=share_link)  |
| Video Deblurring                     | DVD   | Ours-s | [gdrive](https://drive.google.com/file/d/181FLpwRe87iFa-7U3myxR3nNc8qjRe-z/view?usp=share_link)  |
| Video Denoising                    | DAVIS & Set8| Ours+ | [gdrive](https://drive.google.com/file/d/1zCKZ5t7qRTTEt8_loR_Fa4ODpNOTodyM/view?usp=share_link) |
| Video Denoising                    | DAVIS & set8 | Ours-s | [gdrive](https://drive.google.com/file/d/1SJBxI2wOwgHAWx5h2N02-1UpIvEKalCg/view?usp=share_link)  |



### Visual Results

| Task                                 | Dataset | Model      | Visual Results  |
| :----------------------------------- | :------ | :----------- | :------------------------ |
| Video Deblurring                     | GoPro   | Ours+ | [gdrive](https://drive.google.com/file/d/12wJEndw6iJtt4Vg3BeRq7G3bEewPg-Vd/view?usp=share_link) |
| Video Deblurring                     | GoPro   | Ours-s | [gdrive](https://drive.google.com/file/d/17ybazJfVpiH_RyZJtQOuzVdX3V4EdziL/view?usp=share_link) |
| Video Deblurring                     | DVD   | Ours+ | [gdrive](https://drive.google.com/file/d/1GeQPFCHiwwwBgCnrRWFD4rR7BRyX8KVi/view?usp=share_link)  |
| Video Deblurring                     | DVD   | Ours-s | [gdrive](https://drive.google.com/file/d/1R-cZj5SdrtJdgyin8fZbmunS2or6H88M/view?usp=share_link)  |
| Video Denoising                    | DAVIS   | Ours+ | [gdrive](https://drive.google.com/file/d/1HvKtdRr8qUDAK_ijFHQ6YxQ8X_dJaqZj/view?usp=share_link) |
| Video Denoising                    | DAVIS   | Ours-s | [gdrive](https://drive.google.com/file/d/15CgIFQhxAR79ovbtcpBEU1XBlfaOSdlp/view?usp=share_link)  |
| Video Denoising                    | Set8   | Ours+ | [gdrive](https://drive.google.com/file/d/1xm8EnCZqqjSciJAzmIphixUlZUqjioqP/view?usp=share_link) |
| Video Denoising                  | Set8   | Ours-s | [gdrive](https://drive.google.com/file/d/11x6IxwIEHPzCVAhiYOD41rs6yerQbzgV/view?usp=share_link) |


### Citations

```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Dasong and Shi, Xiaoyu and Zhang, Yi and Cheung, Ka Chun and See, Simon and Wang, Xiaogang and Qin, Hongwei and Li, Hongsheng},
    title     = {A Simple Baseline for Video Restoration With Grouped Spatial-Temporal Shift},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9822-9832}
}
```
