# PolaForge: Lightweight Polarization-Aware Network for Multimodal Industrial Defect Detection

## Requirements

### Environment requirements: 

- CUDA 12.1

- Python 3.9

### Dependency requirements: 

- numpy 1.24.0
- thop 0.1.1
- tqdm 4.65.2
- opencv-python 4.10.0.84
- einops 0.8.0
- torch  2.1.2
- torchaudio 2.1.2
- torchvision 0.16.2
- mamba-ssm 1.2.0
- mmcv-full==1.7.2

## Installation

We recommend you to use Anaconda to create a conda environment:

```
conda create -n name python=3.9 pip
```

Then, activate the environment:

```
conda activate name
```

Install the torch, torchvision and torchaudio

``````
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
``````

Now, you can install other requirements:

``````
pip install -r requirements.txt
``````

``````
python main.py
``````


## POL dataset

The TUT dataset is available at [POL](https://github.com/Karl1109/TUT)

## Contact

Any questions, please contact the email at smf@stud.tjut.edu.cn
