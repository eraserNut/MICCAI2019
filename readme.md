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

#### Task1(Nasopharynx cancer):

Organ-at-risk segmentation from head & neck CT scans.

<sub>22 OARs of 50 nasopharynx cancer patients will be annotated and released to public as the training data. Each of the annotated CT scan is marked by one experienced oncologist and verified by another experienced one. 
Another 10 patients’ CT scans will be used as the test data.</sub>

<table>
    <tr>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/acd93612.png" width="150" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/e3c5b158.png" width="150" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/b879fd0b.png" width="150" height="170"/></td>
    </tr>
</table>

| <sub>Model</sub> | <sub>Dice</sub> | <sub>HD95%</sub> | <sub>Encoder</sub> | <sub>Batch size</sub>|<sub>Loss function</sub>| <sub>Resized</sub>|<sub>Use pretrained </sub>|<sub>Rcf refine</sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>FPN</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>DeepLab-V3</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|

#### Task2(Nasopharynx cancer):

Gross Target Volume segmentation of nasopharynx cancer.

<sub>The 50 GTV annotations of the same 50 nasopharynx cancer patients’ CT scans will be provided as the training data and another 10 patients’ GTV will be used as the test data.</sub>

<table>
    <tr>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/81824b0c.png" width="150" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/245a51f1.png" width="150" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/c3f373bd.png" width="150" height="170"/></td>
    </tr>
</table>

| <sub>Model</sub> | <sub>Dice</sub> | <sub>HD95%</sub> | <sub>Encoder</sub> | <sub>Batch size</sub>|<sub>Loss function</sub>| <sub>Resized</sub>|<sub>Use pretrained </sub>|<sub>Rcf refine</sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>FPN</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>DeepLab-V3</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|

#### Task3(Lung cancer):

Organ-at-risk segmentation from chest CT scans.

<sub>6 OARs of 50 lung cancer patients will be annotated and released to public as the training data. Another 10 patients’ CT scans will be used as the test data. Each CT scan is annotated by one experienced oncologist and verified by another one.</sub>

<table>
    <tr>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/9bb6cffc.png" width="220" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/d5a47356.png" width="220" height="170"/></td>
    </tr>
</table>


| <sub>Model</sub> | <sub>Dice</sub> | <sub>HD95%</sub> | <sub>Encoder</sub> | <sub>Batch size</sub>|<sub>Loss function</sub>| <sub>Resized</sub>|<sub>Use pretrained </sub>|<sub>Rcf refine</sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>FPN</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>DeepLab-V3</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>OhemCrossEntropy</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|

#### Task4(Lung cancer):

Gross Target Volume segmentation of lung cancer.

<sub>The 50 GTV annotations of the same 50 lung cancer patients’ CT scans will be provided as the training data and another 10 patients’ GTV will be used as the test data. Each CT scan is annotated by one experienced oncologist and verified by another one.</sub>

<table>
    <tr>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/7795aea7.png" width="250" height="170"/></td>
        <td><img src="https://structseg2019.grand-challenge.org/media/i/d86db96d.png" width="250" height="170"/></td>
    </tr>
</table>

| <sub>Model</sub> | <sub>Dice</sub> | <sub>HD95%</sub> | <sub>Encoder</sub> | <sub>Batch size</sub>|<sub>Loss function</sub>| <sub>Resized</sub>|<sub>Use pretrained </sub>|<sub>Rcf refine</sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>FPN</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|
|<sub>DeepLab-V3</sub>| <sub></sub> | <sub></sub> | <sub>Resnet-18</sub>|<sub>4</sub>|<sub>BCE,Dice</sub>|<sub>N</sub>|<sub>Y</sub>|<sub>N</sub>|