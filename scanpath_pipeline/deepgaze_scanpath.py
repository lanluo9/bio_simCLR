# python deepgaze_scanpath.py --parallel_id 0 --work_dir '/your_working_directory/'

import numpy as np
import math
from scipy.ndimage import zoom
from scipy.special import logsumexp
from tqdm import tqdm

import argparse
import sys
import os
import shutil
import glob
import gc

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='deepgaze scanpath predict')
parser.add_argument('--parallel_id', default=0, type=int,
                    help='run 6 parallel processes. parallel_id determines counter range',
                    choices=[0, 1, 2, 3, 4, 5])
parser.add_argument('--work_dir', default=os.getcwd(), 
                    type=str, help='where to run py & save scanpath result')
args = parser.parse_args()
print(f'parallel id is {args.parallel_id}\n')
print(f'working from directory {args.work_dir}\n')

dir_now = args.work_dir
os.chdir(dir_now)

os.system('wget https://github.com/matthias-k/DeepGaze/archive/refs/heads/main.zip')
shutil.unpack_archive('main.zip', dir_now)
os.remove('main.zip')
sys.path.append(os.path.join(dir_now, 'DeepGaze-main'))
import deepgaze_pytorch
print('imported deepgaze_pytorch\n')

try:
  centerbias_template = np.load('centerbias_mit1003.npy') # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
except:
  os.system('wget https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy')
  centerbias_template = np.load('centerbias_mit1003.npy')
print('loaded centerbias\n')

DEVICE = 'cuda'
model_deepgaze2 = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
model_deepgaze3 = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
print('got models\n')


def deepgaze2_pred(image, model=model_deepgaze2):
  '''use deepgaze 2e to predict a fixation distribution (without fixation history input)'''
  centerbias = zoom(centerbias_template, \
                    (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), \
                    order=0, mode='nearest') # rescale to match image size
  centerbias -= logsumexp(centerbias) # renormalize log density
  centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

  image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
  log_density_prediction = model(image_tensor, centerbias_tensor) # predicted log density for the next fixation location 

  del model, centerbias_tensor, image_tensor
  gc.collect()
  torch.cuda.empty_cache()
  
  return log_density_prediction


def draw_fix_from_pred(log_density_prediction, nfix=1):
  '''draw 4 fixations to fake a fixation history'''

  fix_dist = log_density_prediction.detach().cpu().numpy()[0, 0]
  fix_dist = np.exp(fix_dist)
  assert math.isclose(fix_dist.sum(), 1) # validate 2d dist of prob sum to 1

  flat = fix_dist.flatten() # sample from fix_dist
  sample_index = np.random.choice(a=flat.size, p=flat, size=nfix)  # sample an index from the 1D array with the probability distribution from the original array
  adjusted_index = np.unravel_index(sample_index, fix_dist.shape) # Take this index and adjust it so it matches the original array
  fixations_x = adjusted_index[1]
  fixations_y = adjusted_index[0] # height = axis 0 = y

  return fixations_x, fixations_y


def deepgaze3_pred(image, fixation_history_x, fixation_history_y, model=model_deepgaze3, nfix_total=20):
  '''
  feed fake fixation history to deepgaze 3 to simulate rest of the scanpath
  use log_density_prediction to draw next fixation, update log_density_prediction, until reach nfix_total
  arg:
    image: np.array of shape (xpix, ypix, channel)
    fixation_history_x
    fixation_history_y
    nfix_total: total fixation needed per image, including 4 steps of fake fixation history
  return:
    fixation_history_x
    fixation_history_y
    log_density_prediction: pred prob of next fix
  '''
  centerbias = zoom(centerbias_template, \
                    (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), \
                    order=0, mode='nearest') # rescale to match image size
  centerbias -= logsumexp(centerbias) # renormalize log density
  centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

  image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE) 
  x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
  y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
  log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

  nstep = nfix_total - 4 # subtract 4 steps of fake history, simulate rest of the fixations
  for i in range(nstep):

    fixations_x, fixations_y = draw_fix_from_pred(log_density_prediction, nfix=1) # predict next fixation
    fixation_history_x = np.append(fixation_history_x, fixations_x)
    fixation_history_y = np.append(fixation_history_y, fixations_y)
    
    x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

  del model, centerbias_tensor, image_tensor, x_hist_tensor, y_hist_tensor
  gc.collect()
  torch.cuda.empty_cache()

  return fixation_history_x, fixation_history_y, log_density_prediction


print('defined functions, starting model pred\n')
os.chdir(dir_now)
dataset = datasets.STL10("Dataset", split="unlabeled", download=True, \
                         transform=transforms.ToTensor(),) # download takes 2-4 min
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, \
                        drop_last=False) #TODO: batch size back to larger (75)
nfix_total = 20
scanpath_arr = np.zeros((len(dataset), nfix_total, 2)) # n img, n fix per img, x & y

counter = 0
counter_start = (len(dataset)//6 - 1) * args.parallel_id
counter_end = (len(dataset)//6 - 1) * (args.parallel_id + 1)
print(f'start from img {counter_start}, end on but not including img {counter_end}\n')
npy_files = glob.glob("*.npy")

for images, _ in tqdm(dataloader): # print(images.shape) # singleton, chan, x, y
  scanpath_1_filename = f"stl10_unlabeled_scanpath_deepgaze_{counter}.npy"
  if (counter < counter_start) or (scanpath_1_filename in npy_files):
    print(f'\nskip existing file or out of range counter {counter}\n')
    counter += 1
    continue

  img_large = F.interpolate(images.to('cuda'), [256, 256]) # interpolate to increase scanpath pred perf, 256 seems best
  img_large = torch.transpose(img_large, 1, 3)
  img_large = torch.transpose(img_large, 1, 2) # print(img_large.shape) # batch_size x height x width x 3 

  images_np = img_large.cpu().detach().numpy() # tensor to np
  images_np = np.squeeze(images_np) # print(images_np.shape, type(images_np)) # height x width x 3
  # plt.matshow(images_np[:,:,0]) # ensure img is standing upright

  log_density_prediction = deepgaze2_pred(images_np)
  fixations_x, fixations_y = draw_fix_from_pred(log_density_prediction, nfix=4)
  del img_large, log_density_prediction; torch.cuda.empty_cache()
  fixation_history_x, fixation_history_y, log_density_prediction = deepgaze3_pred(images_np, \
                                                                                  fixations_x, fixations_y, \
                                                                                  nfix_total=nfix_total)
  scanpath_arr[counter,:,0] = fixation_history_x # width
  scanpath_arr[counter,:,1] = fixation_history_y # height
  scanpath_1 = np.squeeze(scanpath_arr[counter,:,:])
  np.save(scanpath_1_filename, scanpath_1) # save each img scanpath as a separate npy file (checked if exist) in case .py get interrupted
  print(f'\nsaved scanpath of img no.{counter}\n')

  gc.collect()
  torch.cuda.empty_cache()

  counter += 1
  if counter >= counter_end: # result doesn't contain endpoint, only contain start point
    break

np.save(f"stl10_unlabeled_scanpath_deepgaze_range_{counter_start}_{counter_end}.npy", scanpath_arr)
print('saved scanpath array of all images in range of counter\n')