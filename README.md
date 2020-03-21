
# TIN+: Transferable Interactiveness Network with Pairwise Body Part Hints             
Improving the detection results of **CVPR2019** paper *"[Transferable Interactiveness Knowledge for Human-Object Interaction Detection](https://arxiv.org/abs/1811.08264)"*  with hints from **Pairwise Body Parts** detection described in **ECCV2018** paper "[Pairwise Body-Part Attention for Recognizing Human-Object Interactions](https://arxiv.org/abs/1807.10889)". The original implementation of **TIN** (Transferable Interactiveness Network) can be seen [Here](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network). 

The implementation for **TIN+** ,the revised **TIN** (Transferable Interactiveness Network) is created by **Haoping Chen**, SJTU. We only do the experiments on **HICO-DET** Dataset.

Contact me in email: apple_chen@sjtu.edu.cn



## Introduction
#### TIN: Transferable Interactiveness Network

Interactiveness Knowledge indicates whether human and object interact with each other or not. It can be learned across HOI datasets, regardless of HOI category settings. We exploit an Interactiveness Network to learn the general interactiveness knowledge from multiple HOI datasets and perform Non-Interaction Suppression before HOI classification in inference. On account of the generalization of interactiveness, our **TIN** is a transferable knowledge learner and can be cooperated with any HOI detection models to achieve desirable results. *TIN* outperforms state-of-the-art HOI detection results by a great margin, verifying its efficacy and flexibility.

![Overview of Our Framework](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/blob/master/images/overview.jpg?raw=true)



#### **Partpair: Pairwise Body Part Network**

Different body parts should be paid with different attention in HOI recognition, and the correlations between different body parts should be further considered. This is because our body parts always work collaboratively. We propose a new pairwise body-part attention model which can learn to focus on crucial parts, and their correlations for HOI recognition. A novel attention based feature selection method and a feature representation scheme that can capture pairwise correlations between body parts are introduced in the model.

![Overview of Pairwise Body Parts](https://github.com/Imposingapple/Transferable_Interactiveness_Network_with_Partpair/blob/master/images/partpair.png?raw=true)

#### Why combining TIN with Partpair

Although TIN has a better overall performance than Partpair, we think Partpair can give some hints for some of Human-Object actions with strong body-pair relations. Just check the detection results of the following picture.

![example for body partpair hint](https://raw.githubusercontent.com/Imposingapple/Transferable_Interactiveness_Network_with_Partpair/master/images/bodypair_guidance.jpg?raw=true)

The confidence for different Human Object actions in one Human Object pair in **TIN** and **Partpair** defers:

| Method        | carry      | hold       | jump       | hop-on     | park       | push       | repair     | ride       |
| :------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Partpair Only | 0.0010     | **0.7488** | 0.0364     | 0.0118     | 0.0116     | 0.0096     | 0.0018     | **0.9089** |
| TIN Only      | **0.0588** | 0.4357     | **0.2455** | **0.2117** | **0.1179** | **0.0771** | **0.0724** | 0.6253     |

Note that here we use the best result for **TIN Only** and **Partpair Only** methods respectly, where we use exactly same ground truth label and loss function. We can see partpair generate very high score on some actions where there's a strong body-pair guidance.



#### Two modes to combine

1. **Joint** Mode: modify the network structure(see folder ./lib/), add the loss from TIN and Partpair together and try to minimize the total loss in training stage.
2. **Separate** Mode: no modify for network structure, we only rely on detection results of best TIN and best Partpair model.

Note that the test output for each Human-Object Pair is two 600-dimensional vector in both Joint and Separate Mode, one from TIN and one from Partpair. The experiment results listed below show that Separate Mode has a better overall performance, which is because the Partpair Network is overfitting when TIN  generates the best model.



## Results on HICO-DET

**Our Results on HICO-DET dataset**

|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:----|:---:|:---:|:---:|:---:|:---:|:---:|
|Partpair Only| 14.37 | 10.23 | 15.61 | 16.65 | 13.43 | 17.61 |
|TIN  Only| 17.54  | 13.80 | 18.65 | 19.75 | 15.70 |20.96|
|TIN && Partpair, max      (Separate)| 16.41 | 13.75 | 17.20 | 18.46 | 15.65 |19.30|
|TIN && Partpair, 0.1 ave         (Separate)| 17.75 | **13.82** | 18.93 | 19.88 | 15.71 |21.13|
|TIN && Partpair, 0.2 ave        (Separate)| **17.81** | 13.79 | **19.01** | **19.89** | 15.71 |**21.14**|
|TIN && Partpair, 0.3  ave      (Separate)| 17.75 | 13.73 | 18.94 | 19.82 | **15.72** |21.04|
|TIN && Partpair, 0.2  ave   (Joint)| 15.40 | 10.52 | 16.86 | 17.36 | 12.55 |18.80|

**Please note that: **

1. All '**TIN**' in the table above refers to **TIN  RP<sub>T2</sub>C<sub>D</sub>(optimized)**. we have reimplemented TIN (e.g. replacing the vanilla HOI classifier with **iCAN** and using cosine_decay lr), thus the result here is different and slight better than the one in [[Arxiv]](https://arxiv.org/abs/1811.08264).

2. **0.x ave** means each of the 600-dimensional HOI actions confidence is fused by:  (score of TIN+ 0.x*score of Partpair) / (1+0.x). **max** means that each of the 600-dimensional HOI actions confidence is determined by max (score of TIN, score of Partpair)

   

## Getting Started

### Installation

1. Clone this repository.

```
git clone https://github.com/Imposingapple/Transferable_Interactiveness_Network_with_Partpair.git
```

2. Download pkl files containing Partpair best detection results, best binary score for NIS, and setup evaluation and API. (The detection results (person and object boudning boxes) are collected from: iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/).)

```
chmod +x ./script/Dataset_download.sh 
./script/Dataset_download.sh
```

​	Note that if you can't download these files, you may check if you can connect the Google Drive or you can 	directly download files by inserting the following code on you web browser: 

```
https://docs.google.com/uc?export=download&id={id}
```

​	and then put them in correct folder. The *id* you may insert and the *correct folder* of each file can be seen 	just in Dataset_download.sh. 

3. Download HICO dataset

```
https://docs.google.com/uc?export=download&id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk
```

​	Unzip the dataset and put it in ./Data/

4. Install Python dependencies.

   If you have trouble installing requirements, try to update your pip or try to use conda.

### Training

1.Train on HICO-DET dataset(Joint)

```
python tools/Train_HICO.py --num_iteration 2000000 --model TIN_partpair
```

Note that in **separate** mode, we do not need to train the two network together, we just need to use the best detection results of TIN and Partpair and find wise way to combine them.

### Testing

1.Test on HICO-DET dataset(Joint)

```
python tools/Test_HICO_Joint.py --num_iteration 1700000 --model TIN_partpair
```

2.Test on HICO-DET dataset(Separate)

```
python tools/Test_HICO_Separate.py --num_iteration 1700000 --model TIN_partpair
```

## Acknowledgement

Some of the codes are built upon iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/ ). Thanks them for their great work! The pose estimation results are obtained from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) . **Alpha Pose** is an accurate multi-person pose estimator, which is the **first real-time** open-source system that achieves **70+ mAP (72.3 mAP)** on COCO dataset and **80+ mAP (82.1 mAP)** on MPII dataset. You may also use your own pose estimation results to train the interactiveness predictor, thus you could directly donwload the train and test pkl files from iCAN [[website]](http://chengao.vision/iCAN/) and insert your pose results.

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

**TIN+** (Transferable Interactiveness Network with Pairwise Body Part Hints) is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. I will send the detail agreement to you.
