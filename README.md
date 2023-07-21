# Subject-Diffusion


[[Project Page](https://oppo-mente-lab.github.io/subject_diffusion/)] [[Paper](https://arxiv.org/abs/2303.17870)]


## Requirements
A suitable [conda](https://conda.io/) environment named `subject-diffusion` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate subject-diffusion
```

## Data Prepare 
First, you need install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/). Then run:
```bash
python data_process.py tar_path tar_index_begin tar_index_end output_path
```
The first parameter represents the data path of webdataset image text pair. The original data can be downloaded by [img2dataset](https://github.com/rom1504/img2dataset) command; The last two parameters represent the beginning and end of the index for webdataset data

## Training 


```bash
bash train.sh 0 8
```
The first parameter represents the global rank of the current process, used for inter process communication. The host with rank=0 is the master node.
and the second parameter is the world size. Please review the detailed parameters of model training
with train_en.sh script

## Inference

We provide a script to generate images using pretrained checkpoints. run
```bash
python test.py
```


## TODOs

- [x] Release inference code
- [x] Release training code
- [x] Release data preparation code
- [ ] Release demo


## Acknowledgements
This repository is built on the code of [diffusers](https://github.com/huggingface/diffusers) library.
Additionally, we borrow some code from [GLIGEN](https://github.com/gligen/GLIGEN), [FastComposer](https://github.com/mit-han-lab/fastcomposer) and [GlyphDraw](https://github.com/OPPO-Mente-Lab/GlyphDraw).