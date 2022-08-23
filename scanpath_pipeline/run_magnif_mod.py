# python run_magnif_mod.py -data /content/Dataset -dataset-name stl10 \
#     --workers 16 --log_root /content/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
#     --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
#     --run_label test  --magnif \
#     --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
#     --gridfunc_form radial_quad  --sample_temperature 1.5  --sampling_bdr 16 \
#     --K 20  --temperature 0.07

import argparse
import sys
import os
import shutil
import glob
import gc

dir_now = '/content/'
os.chdir(dir_now)

print('\ndownload Foveated_Saccade_SimCLR\n')
os.system('wget https://github.com/Animadversio/Foveated_Saccade_SimCLR/archive/refs/heads/dev.zip')
shutil.unpack_archive('dev.zip', dir_now)
os.remove('dev.zip')
sys.path.append(os.path.join(dir_now, 'Foveated_Saccade_SimCLR-dev'))

print('\ndownload revision in bio_simclr\n')
os.system('wget https://github.com/lanluo9/bio_simCLR/archive/refs/heads/master.zip')
shutil.unpack_archive('master.zip', dir_now)
os.remove('master.zip')

print('\nreplacing revised files\n')
file_orig_dir = os.path.join(dir_now, 'bio_simCLR-master/scanpath_pipeline/')
file_destination = os.path.join(dir_now, 'Foveated_Saccade_SimCLR-dev/data_aug')
shutil.move(os.path.join(file_orig_dir, 'cort_magnif_tfm.py'), os.path.join(file_destination, 'cort_magnif_tfm.py'))
shutil.move(os.path.join(file_orig_dir, 'dataset_w_salmap.py'), os.path.join(file_destination, 'dataset_w_salmap.py'))

file_destination = os.path.join(dir_now, 'Foveated_Saccade_SimCLR-dev/')
shutil.move(os.path.join(file_orig_dir, 'simclr.py'), os.path.join(file_destination, 'simclr.py'))

print('\npip install kornia & gdown\n')
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("kornia")
install("gdown")
import gdown

print('\ndownload predicted scanpath\n')
data_path = os.path.join(dir_now, 'Dataset')
if not os.path.exists(data_path):
  os.mkdir(data_path)
os.chdir(data_path)
url = "https://drive.google.com/uc?id=1t5ka0_gEQSxjgDxZclw_y0eZffUb8M0r"
output = "stl10_unlabeled_scanpath_deepgaze.npy" # added inhibition of return
gdown.download(url, output, quiet=False)

print('\ndownload predicted saliency map\n')
url = "https://drive.google.com/uc?id=1cXp7Qg0O23lGyYnjS1a7oUCOY8hw3ckn"
# url = "https://drive.google.com/uc?id=1UybYg2VkZcO5q4Z4Y5lWJ12q0sG88-Hk"
output = "stl10_unlabeled_salmaps_salicon.npy"
gdown.download(url, output, quiet=False)

os.chdir(os.path.join(dir_now, 'Foveated_Saccade_SimCLR-dev'))
print(f'\ncurrent directory: {os.getcwd()}\n')

############################################## below is original run_magnif.py

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from os.path import join
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--randomize_seed', action='store_true', default=False,
                    help='Set randomized seed for the experiment')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--ckpt_every_n_epocs', default=100, type=int,
                    help='Log every n epocs')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


parser.add_argument('--log_root', default="/scratch1/fs1/crponce/simclr_runs", \
    type=str, help='root folder to put logs')
parser.add_argument('--run_label', default="", \
    type=str, help='folder prefix to identify runs')


parser.add_argument('--crop', action='store_true', default=False, help='Enable crop')
parser.add_argument('--disable_blur', action='store_true', default=False,  # blur == True
    help='Do Deperministic Gaussian blur augmentation ')

parser.add_argument('--magnif', action='store_true', default=False,
    help='Do random magnif augmentation')
parser.add_argument('--sal_sample', action='store_true', default=False,
    help='Use saliency map to guide sampling or not')
parser.add_argument('--sal_control', action='store_true', default=False,
    help='Flat density as control')
parser.add_argument('--sample_temperature', default=1.5, \
    type=float, help='temperature of sampling ')


parser.add_argument('--gridfunc_form', default='radial_quad', type=str, choices=['radial_exp', 'radial_quad'],
    help='Formula for the grid function')
parser.add_argument('--sampling_bdr', default=16,
    type=int, help='border width for sampling the fixation point on the image')
parser.add_argument('--cover_ratio', default=(0.05, 0.7),
    type=float, nargs="+", help='Range of fovea area as a ratio of the whole image size.')

parser.add_argument('--fov_size', default=20,
    type=float, help='Scaling coefficent for kernel of foveation blur')
parser.add_argument('--K', default=20,
    type=float, help='border width for sampling the fixation point on the image')

parser.add_argument('--slope_C', default=1.5,
    type=float, nargs="+", help='Scaling of the exponential radial function, controlling the degree of distortion introduced by '
                                'the transform; usually in [0.75, 3.0], 0.5 will be not distorted; '
                                '3.0 will be highly distorted. It can be a range to randomlize uniformly. ')

parser.add_argument('--dry_run', action='store_true', default=False,  # blur == True
    help='If this flag is true, then stop before training really starts. Use this to test the augmentation and arguments. ')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.blur = not args.disable_blur
    if type(args.slope_C) in [list, tuple] and len(args.slope_C) == 1:  # make it a scaler
        args.slope_C = args.slope_C[0]

    if type(args.cover_ratio) in [list, tuple] and len(args.cover_ratio) == 1:  # make it a scaler
        args.cover_ratio = args.cover_ratio[0]

    print(args)

    # from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
    # dataset = ContrastiveLearningDataset(args.data)
    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    from data_aug.dataset_w_salmap import Contrastive_STL10_w_salmap, Contrastive_STL10_w_CortMagnif
    from data_aug.saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density, RandomResizedCrop
    from data_aug.cort_magnif_tfm import get_RandomMagnifTfm
    from data_aug.visualize_aug_dataset import visualize_augmented_dataset

    train_dataset = Contrastive_STL10_w_CortMagnif(dataset_dir=args.data,
            split="unlabeled", crop=args.crop, magnif=args.magnif, 
            sal_sample=args.sal_sample, sal_control=args.sal_control)
    train_dataset.transform = train_dataset.get_simclr_pre_magnif_transform(96,
                        blur=args.blur, crop=args.crop, )
    # train_dataset.transform = train_dataset.get_simclr_magnif_transform(96,
    #                     blur=args.blur, crop=args.crop, magnif=args.magnif,
    #                     sal_sample=args.sal_sample, sample_temperature=args.sample_temperature,
    #                     gridfunc_form=args.gridfunc_form, bdr=args.sampling_bdr,
    #                     fov=args.fov_size, K=args.K, cover_ratio=args.cover_ratio,
    #                     slope_C=args.slope_C, )
    if args.magnif:
        if args.gridfunc_form == "radial_quad":
            train_dataset.magnifier = get_RandomMagnifTfm(grid_generator="radial_quad_isotrop",
                                bdr=args.sampling_bdr, fov=args.fov_size, K=args.K, cover_ratio=args.cover_ratio,
                                sal_sample=args.sal_sample, sample_temperature=args.sample_temperature,)
        elif args.gridfunc_form == "radial_exp":
            train_dataset.magnifier = get_RandomMagnifTfm(grid_generator="radial_exp_isotrop",
                                bdr=args.sampling_bdr, slope_C=args.slope_C, cover_ratio=args.cover_ratio,
                                sal_sample=args.sal_sample, sample_temperature=args.sample_temperature,)
        else:
            raise ValueError
    else:
        train_dataset.magnifier = None

    if args.randomize_seed:
        seed = torch.random.seed()
        args.seed = seed
        print("Use randomized seed to test robustness, seed=%d" % seed)
    else:
        print("Use fixed manual seed, seed=0")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args) # args carry the global config variables here.
        mtg = visualize_augmented_dataset(train_dataset)
        mtg.save(join(simclr.writer.log_dir, "sample_data_augs.png"))  # print sample data augmentations
        if args.dry_run:
            return
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
