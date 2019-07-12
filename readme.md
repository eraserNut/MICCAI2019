# MICCAI 2019 workshop

### Table of content
1. [Architectures](#architectures)
2. [Experiment results](#results)
3. ...

### Architectures <a name="architectures"></a>
- Unet
- FPN
- DeepLab-V3

### Experiment results <a name="results"></a>

<sub>We test these four segmentation models at a liver datasets. We rewrite the framework with reference to [DAF](https://github.com/zijundeng/DAF).
 The model implementations of Unet,FPN and DAF are dependent on github while Unet++ is implemented by ourself. 
 For fair, we train all models with 60 epochs and add evaluation for each epoch. Then save best epoch as the last result. We evaluate them with two norm: Dice and F1, all results as follows:</sub>

#####Task1:
Organ-at-risk segmentation from head & neck CT scans.

<sub>22 OARs of 50 nasopharynx cancer patients will be annotated and released to public as the training data. Each of the annotated CT scan is marked by one experienced oncologist and verified by another experienced one. 
Another 10 patients’ CT scans will be used as the test data.</sub>

![Image text](https://structseg2019.grand-challenge.org/media/i/acd93612.png)
![Image text](https://structseg2019.grand-challenge.org/media/i/e3c5b158.png)
![Image text](https://structseg2019.grand-challenge.org/media/i/b879fd0b.png)

| <sub>Model</sub> | <sub>Dice</sub> | <sub>HD95%</sub> | <sub>Encoder</sub> | <sub>Batch size</sub>|<sub>Loss function</sub>| <sub>Resized</sub>|<sub>Use pretrained </sub>|<sub>Rcf refine</sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| <sub>0.9494</sub> | <sub>0.9503</sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>bcd,dice</sub>|<sub>(448,448)</sub>|<sub>Y</sub>|<sub>Y</sub>|
|<sub>Unet++</sub>| <sub>0.9453</sub> | <sub>0.9465</sub> | <sub>Resnet-18</sub>|<sub>8</sub>|<sub>bcd,dice</sub>|<sub>(224,224)</sub>|<sub>Y</sub>|<sub>Y</sub>|
|<sub>FPN</sub>| <sub>0.9493</sub> | <sub> 0.9491</sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>bcd,dice</sub>|<sub>(448,448)</sub>|<sub>Y</sub>|<sub>Y</sub>|
|<sub>DAF</sub>| <sub>**0.9515**</sub> | <sub>**0.9515**</sub> | <sub>ResNext-101</sub>|<sub>4</sub>|<sub>bcd,dice</sub>|<sub>(448,448)</sub>|<sub>Y</sub>|<sub>Y</sub>|