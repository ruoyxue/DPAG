import os
import random
import argparse

pwd_dir = os.path.dirname(os.path.abspath('.')) # preparation
filelist_dir = os.path.join(pwd_dir, 'data')


def prepare_lrs2_scp(rootdir, dst, name):
    filelist = []
    for split in os.listdir(rootdir):
        for spk in os.listdir(f"{rootdir}/{split}"):
            if os.path.isdir(f"{rootdir}/{split}/{spk}"):
                src_dir = os.path.join(rootdir, split, spk)
                txt_dir = os.path.join(rootdir, split, spk)
                for fn in os.listdir(src_dir):
                    if fn[-3:] == 'mp4':
                        vfn = fn.split('.')[0]
                        dst_dir = f"{dst}/{split}/{spk}"
                        filelist.append(f"{vfn}\t{src_dir}\t{txt_dir}\t{dst_dir}")
    random.shuffle(filelist)
    with open(f'{name}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
        

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='source dir of downloaded files')
parser.add_argument('--dst', required=True, help='dst dir of processed video files')
parser.add_argument('--dataset', required=True)

if __name__== '__main__':
    args = parser.parse_args()
    os.makedirs(f'{args.dst}', exist_ok=True)
    
    if args.dataset == 'lrs2':
        prepare_lrs2_scp(args.src, args.dst, args.dataset)
