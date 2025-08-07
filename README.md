# DDE-CLIP
DDE-CLIP: Detail-Guided Dual-Modal Enhancement for Zero-Shot Anomaly Detection

## Overview
![overview](figure/f2.jpg)

### Prepare Dataset
Download the dataset and put them in ./data/:

[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [SDD](https://www.vicos.si/resources/kolektorsdd/), [RAD](https://drive.google.com/drive/folders/14sTEtptHbhECPbd7WyhpvtR9BA4YdjCP), [CID](https://drive.google.com/drive/folders/1JW2-_LsmwQkYQkZ23zLAD124ZGNOAtkA), [3CAD](https://drive.google.com/file/d/1BIX0H8TZp0wmrAnXPw8_aCAIX1j1Fzwz/view)

Generate dataset json:
```bash
python generate_dataset_json/mvtec.py
```

### Download Pretrained Weight
 Download the CLIP weights pretrained by OpenAI [[ViT-L-14-336.pt](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)] to ./pretrained_weights/

### Create Environments
```bash
conda create -n dde python=3.9
conda activate dde
pip install -r requirements.txt
```

### Test
Run (the weight trained by us is in ./checkpoint/test.pth/):
```bash
bash test.sh
```

### Train
You can train your own weight:
```bash
bash train.sh
```
