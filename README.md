# Engagement Estimation for the Elderly

This repository contains the codes and documents for paper *Engagement Estimation of the Elderly from Wild Multiparty Human-Robot Interaction*.

```txt
@article{
    title   = {Engagement Estimation of the Elderly from Wild Multiparty Human-Robot Interaction},
    author  = {Zhang, Zhijie and Zheng, Jianmin and Thalmann, Nadia Magnenat},
    year    = {2022}
}

Zhijie Zhang
zhijie002@e.ntu.edu.sg
Singapore
Nanyang Technological University

Jianmin Zheng
asjmzheng@ntu.edu.sg
Singapore
Nanyang Technological University

Gauri Tulsulkar
gauri.rt9@gmail.com
Singapore
Nanyang Technological University

Nidhi Mishra
Nidhimishra2906@gmail.com
Switzerland
Universityof Neuchatel

Nadia Magnenat Thalmann
thalmann@miralab.ch
Switzerland
University of Geneva
```

## Run

```bash
python train.py --model Nonlocal_FC3_Reg --data roi_l
python test.py --model Nonlocal_FC3_Reg --data roi_l
```

### Feature Extraction

- [I3D ResNet Feature Extraction](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)
- [Non-local Neural Networks for Video Classification](https://github.com/facebookresearch/video-nonlocal-net)

1. Download pretrained weights for I3D from the nonlocal repo
    ```bash
    wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
    ```
2. Convert weights from caffe2 to pytorch. This is just a simple renaming of the blobs to match the pytorch model
    ```bash
    python3 convert_weights.py
    ```

Details:
- **sampling**: video folder --> videos (interaction sessions) --> video clips (32 frames sampled every 5 frames, about 10.67s) --> extracted images `(32, 224, 224)`
- **feature extraction**: extracted images --> R3D features `(clp_num, 1024, 4, 14, 14)` --> body bounding boxes --> RoI align features `(clp_num, num_p, 1024, 4, 7, 7)`
- [model_print_r3d](logs/model_print_r3d.txt), [model_info_r3d](logs/model_info_r3d.txt)

#### Conv and maxpooling

```txt
 x x 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 x x
|x|x|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|x|x|
|x|x|+|+|+|                                                                --> | .2-03 |  1
    |+|+|+|+|+|                                                            --> | 01-05 |  2
        |+|+|+|+|+|                                                        --> | 03-07 |  3
            |+|+|+|+|+|                                                    --> | 05-09 |  4
                |+|+|+|+|+|                                                --> | 07-11 |  5
                    |+|+|+|+|+|                                            --> | 09-13 |  6
                        |+|+|+|+|+|                                        --> | 11-15 |  7
                            |+|+|+|+|+|                                    --> | 13-17 |  8
                                |+|+|+|+|+|                                --> | 15-19 |  9
                                    |+|+|+|+|+|                            --> | 17-21 | 10
                                        |+|+|+|+|+|                        --> | 19-23 | 11
                                            |+|+|+|+|+|                    --> | 21-25 | 12
                                                |+|+|+|+|+|                --> | 23-27 | 13
                                                    |+|+|+|+|+|            --> | 25-29 | 14
                                                        |+|+|+|+|+|        --> | 27-31 | 15
                                                            |+|+|+|+|x|    --> | 29-33 | 16

 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|+|
|+|+|                              --> | .2-05 | 1
    |+|+|                          --> | 03-09 | 2
        |+|+|                      --> | 07-13 | 3
            |+|+|                  --> | 11-17 | 4
                |+|+|              --> | 15-21 | 5
                    |+|+|          --> | 19-25 | 6
                        |+|+|      --> | 23-29 | 7
                            |+|+|  --> | 27-33 | 8

 1 2 3 4 5 6 7 8
|+|+|+|+|+|+|+|+|
|+|+|              --> | .2-09 | 04
    |+|+|          --> | 07-17 | 12
        |+|+|      --> | 15-25 | 20
            |+|+|  --> | 23-33 | 28
```

### RoI Align

[`torchvision.ops.roi_align()`](https://pytorch.org/vision/main/_modules/torchvision/ops/roi_align.html#roi_align) performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

```txt
roi_align(
    input          : Tensor
    boxes          : Union[Tensor, List[Tensor]]
    output_size    : BroadcastingList2[int]
    spatial_scale  : float = 1.0
    sampling_ratio : int = -1
    aligned        : bool = False
)
```

### Non-local Blcok

- [Non-local PyTorch](https://github.com/AlexHex7/Non-local_pytorch)
- [PyTorch Implementation of Non-Local Neural Network](https://github.com/tea1528/Non-Local-NN-Pytorch)

Details:
- extracted features `(clp_num, num_p, 1024, 4, 7, 7)` --> self-attention module `(N, 4, 1024, 7, 7)`, `N = clp_num x num_p` --> fc `(N, 1024)`
- [model_info_nl](logs/model_info_nl.txt)

### Comparison Models

|            Method             |                   Models                    | Results |
| :---------------------------: | :-----------------------------------------: | ------: |
| Anagnostopoulou2021Engagement |                   AlexNet                   |         |
|   Oertel2021EngagementAware   |                  SVM / DT                   |         |
|      Saleh2021Improving       |                     I3D                     |         |
|    Steinert2020Engagement     |         {OpenFace, VGGFace} + LSTM          |         |
|      Sumer2021Multimodal      |       Attention + Affect (ResNet 50)        |         |
|       Zhu2020Multirate        | {OpenFace, OpenPose, VGG} + GRU + Attention |         |
