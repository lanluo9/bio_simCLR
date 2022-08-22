import argparse
import sys
import os
import shutil
import glob
import gc

dir_now = os.getcwd()
os.chdir(dir_now)

os.system('wget https://github.com/Animadversio/Foveated_Saccade_SimCLR/archive/refs/heads/dev.zip')
shutil.unpack_archive('dev.zip', dir_now)
os.remove('dev.zip')
sys.path.append(os.path.join(dir_now, 'Foveated_Saccade_SimCLR-dev'))
# import deepgaze_pytorch
print('imported deepgaze_pytorch\n')
