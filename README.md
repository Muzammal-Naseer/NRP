# Self-supervised Approach for Adversarial Robustness

Pytorch Implementation of "Self-supervised Approach for Adversarial Robustness" (CVPR 2020) ([arXiv link])(https://..).

### Table of Contents  
1) [Contributions](#Contributions) <a name="Contributions"/>
2) [Claims](#Claims) <a name="Claims"/>
2) [Usage](#Usage) <a name="Usage"/>
3) [Pretrained-Purifier](#Pretrained-Purifier) <a name="Pretrained-Purifier"/>
4) [Citation](#Citation)  <a name="Citation"/>

## Contributions

1) *Self Supervised Perturbations (SSP):* Our adversarial attack perturb a given clean image with random noise and then maximize perceptual feature distance ($l_{2}$ as an example) w.r.t the clean image. This allows transferable task-agnostic adversaries. The proposed attack can be used to evaluate robustness (black-box) of various computer vision systems such object detection, segmentation and classfication etc.
2) *Neural Representation Purification (NRP):* Our defense is then based on training a purifier network that tries to minimize the perceputal feature difference between clean and SSP generated adversary. NRP enjoys follows benefits:
    * NRP does not require access to original data distribution. For example, NRP trained on MS-COCO dataset can successfully defend ImageNet models.
    * NRP does not require label data.

![Learning Algo](/assets/defenseoverview-min3.png)


