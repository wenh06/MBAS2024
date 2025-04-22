# MICCAI MBAS 2024 - Multi-class Bi-Atrial Segmentation Challenge

<!-- toc -->

- [Introduction](#introduction)
- [Description of the files/folders(modules)](#description-of-the-filesfoldersmodules)
- [Deep learning models for medical image studies](#deep-learning-models-for-medical-image-studies)

<!-- tocstop -->

## Introduction

Atrial fibrillation (AF) is the most common form of cardiac arrhythmia and is associated with substantial morbidity and mortality.
Current clinical treatments for AF perform poorly due to a lack of basic understanding of the underlying atrial anatomical structure
which directly sustains AF in the human atria. In recent years, Late gadolinium-enhanced MRI (LGE-MRI) is widely used to study fibrosis/ scarring,
and clinical studies on AF patients using LGE-MRIs have shown that the extent and distribution of atrial fibrosis can be used to
reliably predict ablation success rates. As a result, direct analysis of the atrial structure in patients with AF is vital to improving
the understanding and patient-specific treatment of AF.

Building on the 2018 left atrium challenge, this new challenge broadens to include both left and right atriums as well as their walls,
focusing on multi-class machine learning from LGE-MRIs to enhance ablation for atrial fibrillation patients. It tests methods for segmentation
and biomarker identification (like atrium volume and fibrosis) using 200 multi-center 3D LGE-MRIs—the largest dataset in the field,
with each scan meticulously labeled by three experts. These new AI and clinical methodologies not only play a significant paradigm shift in cardiac analysis
but also have the potential to be applied across various medical domains, aiming to refine ablation strategies for treating persistent atrial fibrillation.

### LG-MRI

to be added....

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project.
- [const.py](const.py): constant definitions, mostly the URLs for downloading the model weights.
- [data_reader.py](data_reader.py): data reader, including data downloading, file listing, data loading, etc.
- [dataset.py](dataset.py): dataset class, which feeds data to the models.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.
- [evaluate-results](evaluate-results): file for recording the evaluation results on the validation set.
- [outputs.py](outputs.py): container (dataclass) for the outputs of the models.
- [pipeline.py](pipeline.py): pipeline for doing the inference, serving as the entry point for the submission.
- [post_docker_build.py](post_docker_build.py): script for post-processing the docker image after building, typically caching trained models.
- [predict.py](predict.py): command line interface for the submission (i.e. for [pipeline.py](pipeline.py)).
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt): requirements files for different purposes.
- [submissions](submissions): file for recording the evaluation results on the hidden test set.
- [trainer.py](trainer.py): trainer class, which trains the models.
- [train_models.ipynb](train_models.ipynb): notebook for training the models.

</details>

### Folders(Modules)

- [models](models): folder for model definitions, currently containing [VNet models](models/vnet.py), and [nested VNet models](models/nested_vnet.py). [Custom loss functions](models/loss) are also included in this folder, but not implemented yet.
- [utils](utils): utility functions for [computing the metrics](utils/scoring_metrics.py), and most importantly, for conducting Multidimensional Contrast Limited Adaptive Histogram Equalization ([MCLAHE](utils/mclahe_tf.py)).

<details>
<summary>Click to view the details</summary>

</details>

## Deep learning models for medical image studies

[Nividia: Visual Foundation Models for Medical Image Analysis](https://developer.nvidia.com/blog/visual-foundation-models-for-medical-image-analysis/)

[MONAI Model Zoo](https://monai.io/model-zoo.html) | [MONAI at GitHub](https://github.com/Project-MONAI)

## Validation phase leaderboard

[Leaderboard](https://docs.google.com/spreadsheets/d/1YJvwBJjli6htgvomZxk--wPbyq8PtTWBbFIQBw-eroE/)
