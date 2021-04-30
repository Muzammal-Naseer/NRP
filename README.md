# Self-supervised Approach for Adversarial Robustness

[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf) ([arXiv link](https://arxiv.org/abs/2006.04924)), [1-min Presentation](https://drive.google.com/file/d/1aXnRaZGcMZFbhIWKe3K6BiisYti75iOe/view?usp=sharing), [5-min Presentation](https://drive.google.com/file/d/1qUSC0KXPqRFtP5QB9Y70_ZKXqfDFmA7W/view?usp=sharing), [Poster](https://drive.google.com/file/d/1jpuXZZIhGpFrWcA8JvEzxqJ6A0kWphG2/view?usp=sharing)

### Table of Contents  
1) [Contributions](#Contributions) <a name="Contributions"/>
2) [Claims](#Claims) <a name="Claims"/>
3) [How to run SSP Attack?](#how-to-run-SSP-Attack)
4) [Pretrained Purifiers](#Pretrained-Purifiers) 
5) [How to purify Adversarial Images?](#How-to-purify-Adversarial-Images) 
6) [How to by-pass NRP using Straight through Estimation?](#How-to-by-pass-NRP-using-Straight-through-Estimation)
7) [NRP as Dynamic Defense](#NRP-as-Dynamic-Defense)
8) [What Can You Do?](#What-Can-You-Do) 
9) [Citation](#Citation)  <a name="Citation"/>
10) [An Example of Purification of Unseen Adversaries](#An-Example-of-Purification-of-Unseen-Adversaries) <a name="Example"/>


## Contributions

1) *Self Supervised Perturbations (SSP):* Our adversarial attack perturb a given clean image with random noise and then maximize perceptual feature distance ($l_{2}$ as an example) w.r.t the clean image. This allows transferable task-agnostic adversaries. The proposed attack can be used to evaluate robustness (black-box) of various computer vision systems such object detection, segmentation and classification etc.
2) *Neural Representation Purification (NRP):* Our defense is then based on training a purifier network that tries to minimize the perceptual feature difference between clean and SSP generated adversary. NRP enjoys follows benefits:
    * NRP does not require access to original data distribution. For example, NRP trained on MS-COCO dataset can successfully defend ImageNet models.
    * NRP does not require label data.

![Learning Algo](/assets/DefenseOverview-min3.png)

## Claims

NRP can be used to defend against adversarial attacks under following threat models.
1) Attacker is unaware of the defense and the backbone model (This is also known as black-box setting).
2) Attacker knows about the defense but unaware of the NRP and backbone model architectures. Attacker trains a local copy of the defense and tries to bypass using e.g. straight through estimator method.
3) **Can NRP defend against white-box attack?** Yes, but with dynamic inference. When attacker has full knowledge of the architecture and pretrained weights of NRP and its backbone model then attacker can by-pass the defense using e.g. straight thought estimation. In such case, we can add random noise before sending the input sample to NRP.

## How to run SSP Attack?
You don't need to worry about labels, just save clean images in a directory and run the following command:
```
  python ssp.py --sourcedir clean_imgs --eps 16 --iters 100
```

## Pretrained Purifiers

Download pretrained purifiers from [here](https://drive.google.com/file/d/1qWqUS9MKGLC5GxQiqqZsw7z72XCSB8Oc/view?usp=sharing) to 'pretrained_purifiers' folder.

These purifiers are based desnet (around 14Million parameters) and ResNet (only 1.2Million parameters) based architecture. They output the purified sample of the same size of input.

## How to purify Adversarial Images?
You can create adversarial images by using [Cross-Domain Attack](https://github.com/Muzammal-Naseer/Cross-domain-perturbations) or any other attack of your choice. Once you have save the adversarial images then run the following command to purify them:

```
  python purify.py --dir adv_images --purifier NRP
```
Purifiers are trained to handle $l_inf <=16$ but you can try $l_2$ bounded attacks as well.

## How to by-pass NRP using Straight through Estimation?
You can by-pass NRP using backpass method. We provide an example of such an attack using targeted PGD:
```
  python bypass_nrp.py --test_dir val/ --purifier NRP --eps 16 --model_type res152
```

## NRP as Dynamic Defense
Dynamic inference can help against whitebox attacks. We use a very simple methodology: Perturbe the incoming sample with random noise and then purify it using NRP. The drawback is that we lose some clean accuracy.

```
  python purify.py --dir adv_images --purifier NRP --dynamic
```


## What Can You Do? 
**Future Resreach:**
You can build on our work with the following objectives in mind:
   * Can you create a blackbox attack (no knowledge of defense or backbone) that is powerful enough to break our defense?
   * Can you create a graybox attack (defense is known but no knowledge of architecture of NRP and its backbone) that can break our defense. We provide two purifier to test such attack. You can use one in your attack and then test on the other one?
   * Can you break our dynamic inference?
   * Can you prepare similar defense for [unrestricted attack](https://arxiv.org/abs/1807.01216)?
   
## Citation
If you find our work, this repository and pretrained purifiers useful. Please consider giving a star :star: and citation.
```
@InProceedings{Naseer_2020_CVPR,
author = {Naseer, Muzammal and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Porikli, Fatih},
title = {A Self-supervised Approach for Adversarial Robustness},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Contact
Muzammal Naseer - muzammal.naseer@anu.edu.au 
<br/>
Suggestions and questions are welcome!


## An Example of Purification of Unseen Adversaries
These adversaris are never seen by NRP during training. First row shows adversarial images while 2nd shows purified images.
![a](/assets/robust_adv.jpg)
![b](/assets/robust_adv_fixed.jpg)
