

### Introduction

This repository is an implementation for *Face Detection of Image Semantic Disentangle Technology* presented in my graudate thesis. In the paper, I propose a novel **A**utoencoder-**C**lassification **G**eometric framework called **ACG** to detect face forgeries. The code is based on Pytorch. Please follow the instructions below to get started.


### Motivation

Briefly, we train a reconstruction network over genuine images only and use the output of the latent feature by the encoder to perform binary classification. Due to the discrepancy in the data distribution between genuine and forged faces, the reconstruction differences of forged faces are obvious and also indicate the probably forged regions. 


### Basic Requirements
Please ensure that you have already installed the following packages.
- [Pytorch](https://pytorch.org/get-started/previous-versions/) 1.7.1
- [Torchvision](https://pytorch.org/get-started/previous-versions/) 0.8.2
- [Albumentations](https://github.com/albumentations-team/albumentations#spatial-level-transforms) 1.0.3
- [Timm](https://github.com/rwightman/pytorch-image-models) 0.3.4
- [TensorboardX](https://pypi.org/project/tensorboardX/#history) 2.1
- [Scipy](https://pypi.org/project/scipy/#history) 1.5.2
- [PyYaml](https://pypi.org/project/PyYAML/#history) 5.3.1

### Dataset Preparation
- I include the dataset loaders for several commonly-used face forgery datasets, *i.e.,* [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html), [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild), and [DFDC](https://ai.facebook.com/datasets/dfdc). You can enter the dataset website to download the original data.
- For FaceForensics++, Celeb-DF, and DFDC, since the original data are in video format, you should first extract the facial images from the sequences and store them. We use [RetinaFace](https://pypi.org/project/retinaface-pytorch/) to do this.

### Config Files
- We have already provided the config templates in `config/`. You can adjust the parameters in the yaml files to specify a training process. More information is presented in [config/README.md](./config/README.md).

### Training
- I use `torch.distributed` package to train the models, for more information, please refer to [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html).
- To train a model, run the following script in your console. 
```{bash}
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 12345 train.py --config path/to/config.yml --pretrain_path path/to/pretrain.bin
```
- `--config`: Specify the path of the config file. 
- `--pretrain`: Specify the path of pretrained model.

### Testing
- To test a model, run the following script in your console. 
```{bash}
python test.py --config path/to/config.yaml
```
- `--config`: Specify the path of the config file.

### Inference
- I provide the script in `inference.py` to help you do inference using custom data. 
- To do inference, run the following script in your console.
```{bash}
python inference.py --bin path/to/model.bin --image_folder path/to/image_folder --device $DEVICE --image_size $IMAGE_SIZE
```
- `--bin`: Specify the path of the model bin generated by the training script of this project.
- `--image_folder`: Specify the directory of custom facial images. The script accepts images end with `.jpg` or `.png`.
- `--device`: Specify the device to run the experiment, e.g., `cpu`, `cuda:0`.
- `--image_size`: Specify the spatial size of input images.
- The program will output the fake probability for each input image like this:
    ```
    path: path/to/image1.jpg           | fake probability: 0.1296      | prediction: real
    path: path/to/image2.jpg           | fake probability: 0.9146      | prediction: fake
    ```
- Type `python inference.py -h` in your console for more information about available arguments.

### Gradcam
- I provide the script in `gradcam.py` to help you do visualization using custom data. 
- To do this, run the following script in your console.
```{bash}
python gradcam.py --model ACG --pth path/to/model.bin --img demo.jpg --save_path save.jpg
```
optional arguments:
```
  -h, --help            show this help message and exit
  --model,  model name, should be exactly same with the file name in /model
  --img,     facial image path
  --save_path,      save activation map path
```
### Result Demo

Origin Image:

![](test_photo/test_img.jpg)

Activation Map:

![](test_photo/test_img_gradcam.jpg)

### Acknowledgement
- I thank prof. Feng Ding for providing the infrastructure and some instructions for me to design this model.
