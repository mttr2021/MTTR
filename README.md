[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12p0jpSx3pJNfZk-y_L44yeHZlhsKVra-?usp=sharing)
[![Open in Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/MTTR/MTTR-Referring-Video-Object-Segmentation)

This repo contains the official implementation of the **CVPR 2022** [paper](https://arxiv.org/abs/2111.14821): 

<div align="center">
<h1>
<b>
End-to-End Referring Video Object Segmentation<br> with Multimodal Transformers
</b>
</h1>
<h4>
<b>
Adam Botach, Evgenii Zheltonozhskii, Dr. Chaim Baskin
    
Technion – Israel Institute of Technology
</b>
</h4>
</div>


https://user-images.githubusercontent.com/29209964/143956960-73e84321-757f-4272-afc5-385900905093.mp4


## Updates
**3/3/2022**

We are excited to announce that our paper was accepted for publication at **CVPR 2022**! 🥳🥳🥳

The paper can be accessed [here](https://arxiv.org/abs/2111.14821).

**8/12/2021**

We listened to your requests and now release interactive demonstrations of MTTR on [Google Colab](https://colab.research.google.com/drive/12p0jpSx3pJNfZk-y_L44yeHZlhsKVra-?usp=sharing) and [Hugging Face Spaces](https://huggingface.co/spaces/MTTR/MTTR-Referring-Video-Object-Segmentation)! 🚀 🤗

We currently recommend using the Colab version of the demonstration as it is a lot faster (GPU accelerated) and has more options. The Spaces demo on the other hand has a nicer interface but is currently much slower since it runs on CPU.

Enjoy! :)


# How to Run the Code
First, clone this repo to your local machine using: 

`git clone https://github.com/mttr2021/MTTR.git`


## Environment Installation
The code was tested on a Conda environment installed on Ubuntu 18.04.
Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

`conda create -n mttr python=3.9.7 pip -y`

`conda activate mttr`

- Pytorch 1.10:

`conda install pytorch==1.10.0 torchvision==0.11.1 -c pytorch -c conda-forge`

Note that you might have to change the cudatoolkit version above according to your system's CUDA version.
- Hugging Face transformers 4.11.3:

`pip install transformers==4.11.3`

- COCO API (for mAP calculations):

`pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

- Additional required packages:

`pip install h5py wandb opencv-python protobuf av einops ruamel.yaml timm joblib`

`conda install -c conda-forge pandas matplotlib cython scipy cupy`


## Dataset Requirements
### A2D-Sentences
Follow the instructions [here](https://kgavrilyuk.github.io/publication/actor_action/) to download the dataset.
Then, extract and organize the files inside your cloned repo directory as follows (note that only the necessary files are
shown):

```text
MTTR/
└── a2d_sentences/ 
    ├── Release/
    │   ├── videoset.csv  (videos metadata file)
    │   └── CLIPS320/
    │       └── *.mp4     (video files)
    └── text_annotations/
        ├── a2d_annotation.txt  (actual text annotations)
        ├── a2d_missed_videos.txt
        └── a2d_annotation_with_instances/ 
            └── */ (video folders)
                └── *.h5 (annotations files) 
```

### JHMDB-Sentences
Follow the instructions [here](https://kgavrilyuk.github.io/publication/actor_action/) to download the dataset.
Then, extract and organize the files inside your cloned repo directory as follows (note that only the necessary files are
shown):

```text
MTTR/
└── jhmdb_sentences/ 
    ├── Rename_Images/  (frame images)
    │   └── */ (action dirs)
    ├── puppet_mask/  (mask annotations)
    │   └── */ (action dirs)
    └── jhmdb_annotation.txt  (text annotations)
```

### Refer-YouTube-VOS
Download the dataset from the competition's website [here](https://competitions.codalab.org/competitions/29139#participate-get_data).

Note that you may be required to sign up to the competition in order to get access to the dataset. 
This registration process is free and short.

Then, extract and organize the files inside your cloned repo directory as follows (note that only the necessary files are
shown):

```text
MTTR/
└── refer_youtube_vos/ 
    ├── train/
    │   ├── JPEGImages/
    │   │   └── */ (video folders)
    │   │       └── *.jpg (frame image files) 
    │   └── Annotations/
    │       └── */ (video folders)
    │           └── *.png (mask annotation files) 
    ├── valid/
    │   └── JPEGImages/
    │       └── */ (video folders)
    │           └── *.jpg (frame image files) 
    └── meta_expressions/
        ├── train/
        │   └── meta_expressions.json  (text annotations)
        └── valid/
            └── meta_expressions.json  (text annotations)
```


## Running Configuration
The following table lists the parameters which can be configured directly from the command line.

The rest of the running/model parameters for each dataset can be configured in `configs/DATASET_NAME.yaml`.

Note that in order to run the code the path of the relevant `.yaml` config file needs to be supplied using the `-c` parameter.

| Command      | Description | 
| :-----------: | :-----------: | 
| -c  | path to dataset configuration file |
| -rm  | running mode (train/eval)|
| -ws  | window size |
| -bs  | training batch size per GPU |
| -ebs  | eval batch size per GPU (if not provided, training batch size is used) |
| -ng  | number of GPUs to run on|


## Evaluation
The following commands can be used to reproduce the main results of our paper using the supplied checkpoint files.

The commands were tested on RTX 3090 24GB GPUs, but it may be possible to run some of them using GPUs with less
memory by decreasing the batch-size `-bs` parameter.

### A2D-Sentences
|Window Size| Command      | Checkpoint File | mAP Result |
|:---:| :-----------: | :-----------: | :-----------: |
|10|`python main.py -rm eval -c configs/a2d_sentences.yaml -ws 10 -bs 3 -ckpt CHECKPOINT_PATH -ng 2`| [Link](https://drive.google.com/file/d/1evEmZyv82vHHpY-chvoVDjUWWW1disun/view?usp=sharing)       | 46.1| |
|8|`python main.py -rm eval -c configs/a2d_sentences.yaml -ws 8 -bs 3 -ckpt CHECKPOINT_PATH -ng 2`| [Link](https://drive.google.com/file/d/1lKKeUwm-GZEAW5c3uOrl7e9bbLSPq28S/view?usp=sharing)       | 44.7| |

### JHMDB-Sentences
The following commands evaluate our A2D-Sentences-pretrained model on JHMDB-Sentences without additional training.

For this purpose, as explained in our paper, we uniformly sample three frames from each video. To ensure proper 
reproduction of our results on other machines we include the metadata of the sampled frames under
`datasets/jhmdb_sentences/jhmdb_sentences_samples_metadata.json`.
This file is automatically loaded during the evaluation process with the commands below.

To avoid using this file and force sampling different frames, change the `seed` and `generate_new_samples_metadata`
parameters under `MTTR/configs/jhmdb_sentences.yaml`.

|Window Size| Command      | Checkpoint File | mAP Result |
|:-----------:| :-----------: | :-----------: | :-----------: |
|10|`python main.py -rm eval -c configs/jhmdb_sentences.yaml -ws 10 -bs 3 -ckpt CHECKPOINT_PATH -ng 2`| [Link](https://drive.google.com/file/d/1evEmZyv82vHHpY-chvoVDjUWWW1disun/view?usp=sharing)       | 39.2| |
|8|`python main.py -rm eval -c configs/jhmdb_sentences.yaml -ws 8 -bs 3 -ckpt CHECKPOINT_PATH -ng 2`| [Link](https://drive.google.com/file/d/1lKKeUwm-GZEAW5c3uOrl7e9bbLSPq28S/view?usp=sharing)       | 36.6 | |

### Refer-YouTube-VOS
The following command evaluates our model on the public validation subset of Refer-YouTube-VOS dataset.
Since annotations are not publicly available for this subset, our code generates a zip file with the predicted masks
under `MTTR/runs/[RUN_DATE_TIME]/validation_outputs/submission_epoch_0.zip`. This zip needs to be uploaded to the
competition server for evaluation. For your convenience we supply this zip file here as well. 

|Window Size| Command      | Checkpoint File | Output Zip| J&F Result|
|:-----------:| :-----------: | :-----------: | :-----------: |:-----------:
|12|`python main.py -rm eval -c configs/refer_youtube_vos.yaml -ws 12 -bs 1 -ckpt CHECKPOINT_PATH -ng 8`|[Link](https://drive.google.com/file/d/1R_F0ETKipENiJUnVwarHnkPmUIcKXRaL/view?usp=sharing)       | [Link](https://drive.google.com/file/d/1ZytzbM3LlQXcA94zPWc059PPasXMGy6u/view?usp=sharing)|55.32|



## Training

First, download the Kinetics-400 pretrained weights of Video Swin Transformer from this [link](https://drive.google.com/file/d/1BF3luuKVTyxve1kFK_2bxtsST2kH7u1P/view?usp=sharing). 
Note that these weights were originally published in video swin's original repo
[here](https://github.com/SwinTransformer/Video-Swin-Transformer).

Place the downloaded file inside your cloned repo directory as
`MTTR/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth`.

Next, the following commands can be used to train MTTR as described in our paper.

Note that it may be possible to run some of these commands on GPUs with less memory than the ones mentioned below
by decreasing the batch-size `-bs` or window-size `-ws` parameters. However, changing these parameters may also affect
the final performance of the model.

### A2D-Sentences
* The command for the following configuration was tested on 2 A6000 48GB GPUs:

|Window Size| Command      | 
|:-----------:| :-----------: |
|10|`python main.py -rm train -c configs/a2d_sentences.yaml -ws 10 -bs 3 -ng 2`| 

* The command for the following configuration was tested on 3 RTX 3090 24GB GPUs:

|Window Size| Command      |
|:-----------:| :-----------: |
|8|`python main.py -rm train -c configs/a2d_sentences.yaml -ws 8 -bs 2 -ng 3`| 

### Refer-YouTube-VOS
* The command for the following configuration was tested on 4 A6000 48GB GPUs:

|Window Size| Command      |
|:-----------:| :-----------: |
|12|`python main.py -rm train -c configs/refer_youtube_vos.yaml -ws 12 -bs 1 -ng 4`| 

* The command for the following configuration was tested on 8 RTX 3090 24GB GPUs.

|Window Size| Command      | 
|:-----------:| :-----------: | 
|8|`python main.py -rm train -c configs/refer_youtube_vos.yaml -ws 8 -bs 1 -ng 8`| 

Note that this last configuration was not mentioned in our paper. 
However, it is more memory efficient than the original configuration (window size 12) while producing a model
which is almost as good (J&F of 54.56 in our experiments).


### JHMDB-Sentences

As explained in our paper JHMDB-Sentences is used exclusively for evaluation, so training is not supported at this time 
for this dataset.

### Citation

Please consider citing our work in your publications if it helped you or if it is relevant to your research:

```
@inproceedings{botach2021end,
  title={End-to-End Referring Video Object Segmentation with Multimodal Transformers},
  author={Botach, Adam and Zheltonozhskii, Evgenii and Baskin, Chaim},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
