
# Neural Visibility Field for Uncertainty-Driven Active Mapping

CVPR 2024

### [Website](https://sites.google.com/view/nvf-cvpr24/) | [Paper](https://arxiv.org/abs/2406.06948/)

[Shangjie Xue](https://xsj01.github.io/), [Jesse Dill](https://www.linkedin.com/in/jesse-n-dill/), [Pranay Mathur](https://matnay.github.io/), [Frank Dellaert](https://dellaert.github.io/), [Panagiotis Tsiotras](https://ae.gatech.edu/directory/person/panagiotis-tsiotras), [Danfei Xu](https://faculty.cc.gatech.edu/~danfei/)

## Installation
NVF has been tested on Ubuntu 20.04 with an RTX 3090, CUDA >= 11.7, python 3.10, and specific versions of nerfacc & viser. Below is line-by-line how we setup our environment dependencies. 
```bash
conda create -y -n nvf python=3.10
conda activate nvf

conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda env config vars set CUDA_INCLUDE_DIRS=$CONDA_PREFIX
conda deactivate; conda activate nvf

python -m pip install --upgrade pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install ninja
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-nvcc
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install nvidia/label/cuda-11.7.1::cuda-cudart

pip install --upgrade pip setuptools
pip install -e .

pip install  git+https://github.com/nerfstudio-project/viser.git@aa417815bf248ba15ee6e22cd4bb49bbc149dee8

pip uninstall -y nerfacc
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@433130618da036d64581e07dc1bf5520bd213129

pip install -r nvf/requirments.txt 
pip install pytorch3d -f  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html

# for image generation
sudo snap install blender --classic

```
## Usage

To evaluate NVF on the hubble scene, run 
```bash
python eval.py --scene hubble --method nvf
```

You can run with different scenes:
```bash
python eval.py --scene {hubble,room,lego,hotdog} --method nvf
```
To train on the datasets, download the .blend and .ply files from our [google drive](https://drive.google.com/drive/folders/1s2g-gBiHMOdQTVtmuTLGfcaA7OMJazvt?usp=sharing) and place them into /data/assets/blend_files/

## Viewing

To view the NVF results during the training process, we use the viewer provided by nerfstudio. After `eval.py` is executing, an http url should be provided in the terminal. To view the viewer over SSH, see nerfstudio's [guide](https://docs.nerf.studio/quickstart/viewer_quickstart.html). By default, the viewer shows the RGB visualization of the scene. To view the uncertainty, select the output render box, it currently should say `rgb`. Choose `entropy` from the dropdown options.

## Configuration

To run NVF and baseline methods in different configurations, see all possible options with 

```bash
python eval.py -h
```

## Codebase

This repo builds upon [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [nerf_bridge](https://github.com/javieryu/nerf_bridge). We use nerfstudio's implementation of Instant-NGP. Our modifications to Instant-NGP can be found in `models/instant_ngp.py`. 

The ground truth training image is rendered through Blender. For adding new scenes for evaluation, take at look at `config.py` and `nvf/env/Scene.py`.

### Citation

If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{xue2024neural,
  title={Neural Visibility Field for Uncertainty-Driven Active Mapping},
  author={Xue, Shangjie and Dill, Jesse and Mathur, Pranay and Dellaert, Frank and Tsiotras, Panagiotis and Xu, Danfei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18122--18132},
  year={2024}
}
```
