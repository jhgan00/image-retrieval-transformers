# Training Vision Transformers for Image Retrieval
 
A (unofficial) PyTorch implementation of [Training Vision Transformers for Image Retrieval](https://arxiv.org/abs/2102.05644).
This code was written for the category-level image retrieval task and tested on CUB-200-2011, Stanford Online Products and In-shop dataset.


<img src="assets/img.png" height="250px">

## Experiments


### Notations

> - IRT<sub>O</sub> – off-the-shelf extraction of features from a ViT backbone, pre-trained on ImageNet;
> - IRT<sub>L</sub> – fine-tuning a transformer with metric learning, in particular with a contrastive loss;
> - IRT<sub>R</sub> – additionally regularizing the output feature space to encourage uniformity.
> - †: Models pre-trained with distillation with a convnet trained on ImageNet1k

<table style="text-align: center">
 <tr>
  <td rowspan="2">method</td>
  <td rowspan="2">backbone</td>
  <td colspan="4">SOP</td>
  <td colspan="4"><a href="http://www.vision.caltech.edu/datasets/cub_200_2011/">CUB-200</a></td>
  <td colspan="4">In-Shop</td>
 </tr>
 <tr>
  <td>1</td>
  <td>10</td>
  <td>100</td>
  <td>1000</td>
  <td>1</td>
  <td>2</td>
  <td>4</td>
  <td>8</td>
  <td>1</td>
  <td>10</td>
  <td>20</td>
  <td>30</td>
 </tr>
 <tr>
  <td>IRT<sub>O</sub></td>
  <td>DeiT-S</td>
  <td>53.13</td>
  <td>68.93</td>
  <td>81.62</td>
  <td>92.77</td>
  <td>58.63</td>
  <td>71.30</td>
  <td>80.93</td>
  <td>88.20</td>
  <td>31.28</td>
  <td>57.03</td>
  <td>64.20</td>
  <td>68.28</td>
 </tr>
 <tr>
  <td>IRT<sub>L</sub></td>
  <td>DeiT-S</td>
  <td>100</td>
  <td>1000</td>
  <td>1</td>
  <td>2</td>
  <td>73.43</td>
  <td>82.77</td>
  <td>88.89</td>
  <td>93.25</td>
  <td>20</td>
  <td>30</td>
  <td>30</td>
  <td>30</td>
 </tr>
 <tr>
  <td>IRT<sub>R</sub></td>
  <td>DeiT-S</td>
  <td>100</td>
  <td>1000</td>
  <td>1</td>
  <td>2</td>
  <td>74.14</td>
  <td>83.15</td>
  <td>89.75</td>
  <td>93.94</td>
  <td>20</td>
  <td>30</td>
  <td>30</td>
  <td>30</td>
 </tr>
 <tr>
  <td>IRT<sub>R</sub></td>
  <td>DeiT-S†</td>
  <td>100</td>
  <td>1000</td>
  <td>1</td>
  <td>2</td>
  <td>76.50</td>
  <td>85.13</td>
  <td>91.12</td>
  <td>94.62</td>
  <td>20</td>
  <td>30</td>
  <td>30</td>
  <td>30</td>
 </tr>
</table>
## Requirements

- python 3.7

```
pip install -r requirements.txt
```

## Training

```bash
# scripts/train.cub.sh
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max_iter 2000 \
  --data-set cub200 \
  --data-path ./data/CUB_200_2011 \
  --rank 1 2 4 8
```

```bash
# scripts/train.sop.sh
python main.py \
  --model deit_small_distilled_patch26_224 \
  --max_iter 35000 \
  --dataset sop \
  --data_path ./data/Stanford_Online_Products \
  --rank 1 10 100 1000
```

```bash
# scripts/train.inshop.sh
python main.py \
  --model deit_small_distilled_patch26_224 \
  --max_iter 35000 \
  --dataset inshop \
  --data_path ./data/In-shop \
  --rank 1 10 20 30
```
