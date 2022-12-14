# -*- coding: utf-8 -*-
"""Copy of Foveate_SSL_Train_Demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1475m4ASae-PMy69MIq9ZnZSguoQdquUR

## Note: You need High RAM runtime to preprocess the STL dataset in memory! (default runtime may run out of memory and crash)
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload 
# %autoreload 2

!pip install kornia

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

# it took 2-5 mins to download the STL10 dataset
dataset = datasets.STL10("Dataset", split="unlabeled", download=True, transform=transforms.ToTensor(),)

"""# Preprocess Dataset with Saliency Models"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/Animadversio/FastSal.git
# %cd FastSal

import model.fastSal as fastsal
from utils import load_weight

# /content/FastSal/model/adaptation_layer.py 
# Line 43: change `misc_nn_ops.Conv2d` to `nn.Conv2d` due to deprecation

model = fastsal.fastsal(pretrain_mode=False, model_type='A') 
# /usr/local/lib/python3.7/dist-packages/torchvision/ops/misc.py:22: 
# FutureWarning: torchvision.ops.misc.Conv2d is deprecated and will be removed in future versions, use torch.nn.Conv2d instead.
# "removed in future versions, use torch.nn.Conv2d instead.", FutureWarning)

state_dict, opt_state = load_weight("weights/salicon_A.pth", remove_decoder=False)
model.load_state_dict(state_dict)
model.cuda().eval();

# colab will crash now due to RAM limit

dataloader = DataLoader(dataset, batch_size=75, shuffle=False, drop_last=False)
salmap_arr = np.zeros((len(dataset), 1, 96, 96), dtype=np.float32)
csr = 0
for images, _ in tqdm(dataloader):
  img_tsr = F.interpolate(images.to('cuda'), [512, 512]) 

  with torch.no_grad():
    salmap = model(img_tsr)

  salmap_small = F.interpolate(salmap, [96, 96]).cpu().numpy()
  csr_end = csr + images.shape[0]
  salmap_arr[csr:csr_end, :, :, :] = salmap_small
  csr = csr_end

np.save("/content/Dataset/stl10_unlabeled_salmaps_salicon.npy", salmap_arr)

!tar -zcvf /content/Dataset/stl10_unlabeled_salmaps_salicon.tar.gz /content/Dataset/stl10_unlabeled_salmaps_salicon.npy

def visualize_salmap():
  figh, axs = plt.subplots(2, 10, figsize=(14, 3))
  for i in range(10):
    idx = np.random.randint(1E5)
    img, _ = dataset[idx]
    salmap = salmap_arr[idx,0,:,:]
    axs[0, i].imshow(img.permute([1,2,0]))
    axs[0, i].axis("off")
    axs[1, i].imshow(salmap)
    axs[1, i].axis("off")
  plt.tight_layout()
  plt.show()
  # figh.savefig("/content/example%03d.png"%np.random.randint(1E3))
visualize_salmap()

"""## Training SimCLR"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
!git clone https://github.com/Animadversio/Foveated_Saccade_SimCLR.git
# %cd /content/Foveated_Saccade_SimCLR
!git checkout dev

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /content/Foveated_Saccade_SimCLR/runs

!python run_magnif.py -data /content/Dataset -dataset-name stl10 --workers 3 --log_root /content/Foveated_Saccade_SimCLR/runs \
	--ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256 \
  --run_label proj256_eval_magnif_bsl --crop &

