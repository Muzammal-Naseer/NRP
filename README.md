# Self-supervised Approach for Adversarial Robustness

Pytorch Implementation of "Self-supervised Approach for Adversarial Robustness" (CVPR 2020) ([arXiv link])(https://..).

### Table of Contents  
1) [Contributions](#Contributions) <a name="Contributions"/>
2) [Claims](#Claims) <a name="Claims"/>
3) [How to run SSP Attack?](#SSP) <a name="SSP"/>
4) [Pretrained-Purifiers](#Pretrained-Purifier) <a name="Pretrained-Purifier"/>
5) [How to purify Adversarial Images?](#purify) <a name="purify"/>
6) [How to by-pass NRP using Straight through Estimation?](#by-pass-NRP)<a name="by-pass-NRP"/>
7) [NRP as Dynamic Defense](#Dynamic-Defense)<a name="Dynamic-Defense"/>
4) [Citation](#Citation)  <a name="Citation"/>

## Contributions

1) *Self Supervised Perturbations (SSP):* Our adversarial attack perturb a given clean image with random noise and then maximize perceptual feature distance ($l_{2}$ as an example) w.r.t the clean image. This allows transferable task-agnostic adversaries. The proposed attack can be used to evaluate robustness (black-box) of various computer vision systems such object detection, segmentation and classfication etc.
2) *Neural Representation Purification (NRP):* Our defense is then based on training a purifier network that tries to minimize the perceputal feature difference between clean and SSP generated adversary. NRP enjoys follows benefits:
    * NRP does not require access to original data distribution. For example, NRP trained on MS-COCO dataset can successfully defend ImageNet models.
    * NRP does not require label data.

![Learning Algo](/assets/DefenseOverview-min3.png)

## Claims

NRP can be used to defend against adversarial attacks under following threat models.
1) Attacker is unaware of the defense and the backbone model (This is also known as black-box setting).
2) Attacker knows about the defense but unaware of the NRP and backbone model architectures. Attacker trains a local copy of the defense and tries to bypass using e.g. straight through estimator method.
3) **Can NRP defend against white-box attack?** Yes, but with dynamic inference. When attacker has full knowledge of the architecture and pretrained weights of NRP and its backbone model then attacker can by-pass the defense using e.g. straight thought estimation. In such case, we can add random noise before sending the input sample to NRP.

## How to run SSP Attack?

## Pretrained-Purifiers

Download pretrained purifiers from [here](https://drive.google.com) to 'saved_models' folder.

These purifiers are based desnet (around 14Million parameters) and ResNet (only 1.2Million parameters) based architecture. They output the purified sample of the same size of input.

## How to purify Adversarial Images?

## How to by-pass NRP using Straight through Estimation?

## NRP as Dynamic Defense

## Citation



