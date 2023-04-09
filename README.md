# CrowdCLIP
An officical implementation of "CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model" (Accepted by CVPR 2023).

# Introduction
Supervised crowd counting relies heavily on costly manual labeling, which is difficult and expensive, especially in dense scenes. To alleviate the problem, we propose a novel unsupervised framework for crowd counting, named CrowdCLIP. The core idea is built on two observations: 1) the recent contrastive pre-trained vision-language model (CLIP) has presented impressive performance on various downstream tasks; 2) there is a natural mapping between crowd patches and count text. To the best of our knowledge, CrowdCLIP is the first to investigate the visionlanguage knowledge to solve the counting problem. Specifically, in the training stage, we exploit the multi-modal ranking loss by constructing ranking text prompts to match the size-sorted crowd patches to guide the image encoder learning. In the testing stage, to deal with the diversity of image patches, we propose a simple yet effective progressive filtering strategy to first select the highly potential crowd patches and then map them into the language space with various counting intervals. Extensive experiments on five challenging datasets demonstrate that the proposed CrowdCLIP achieves superior performance compared to previous unsupervised state-of-the-art counting methods. Notably, CrowdCLIP even surpasses some popular fully-supervised methods under the cross-dataset setting. 
![intro](figs/intro.png)


# Training
Code will be released soon.


## Acknowledgement
Many thanks to the brilliant works ([CLIP](https://github.com/openai/CLIP) and [OrdinalCLIP](https://github.com/xk-huang/OrdinalCLIP))!


## Citation

If you find this codebase helpful, please consider to cite:

```
@article{Liang2023CrowdCLIP,
  title={CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model},
  author={Dingkang Liang, Jiahao Xie, Zhikang Zou, Xiaoqing Ye, Wei Xu, Xiang Bai},
  journal={CVPR},
  year={2023}
} 
```