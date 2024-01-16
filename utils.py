import os
import numpy as np
from glob import glob
import re
import cv2

def read_normalset_calibrate(filedir, filenames, R, resize_ratio=0.25):
    pix_normal = []
    for i in range(len(filenames)):
        data, _ = read_pfm('%s/%s.pfm' % (filedir, filenames[i].split('_')[0]))
        H, W = int(data.shape[0] * resize_ratio), int(data.shape[1] * resize_ratio)
        data = cv2.resize(data, [H, W])
        data = np.matmul(data.reshape(-1, 3), R[i])
        pix_normal.append(data.reshape(H, W, 3))

    return np.array(pix_normal)

def read_imageset(filedir, filenames, resize_ratio=0.25, read_rgb=True):
    pix_normal = []
    for i in range(len(filenames)):
        data = cv2.imread('%s/%02d_000005.png' % (filedir, i))
        if read_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        H, W = int(data.shape[0] * resize_ratio), int(data.shape[1] * resize_ratio)
        data = cv2.resize(data, [H, W])
        pix_normal.append(data.reshape(H, W, 3))

    return np.array(pix_normal)

def read_calib(filename, resize_ratio=1.0):
    calib_list = sorted(glob('%s/*.txt' % filename))
    R, T, K, names = [], [], [], []
    for calib_ in calib_list:
        with open(calib_, 'r') as f:
            lines = f.readlines()
        extrinsic = np.array([i.rstrip().split(' ') for i in lines[1:5]]).astype(np.float32)
        intrinsic = np.array([i.rstrip().split(' ') for i in lines[7:10]]).astype(np.float32)

        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        intrinsic = intrinsic * resize_ratio
        intrinsic = np.concatenate([intrinsic, np.zeros([3, 1])], axis=1)
        intrinsic = np.concatenate([intrinsic, np.zeros([1, 4])], axis=0)
        intrinsic[2, 2] = 0.0
        intrinsic[2, 3] = 1.0
        intrinsic[3, 2] = 1.0

        R.append(rotation)
        T.append(translation)
        K.append(intrinsic)
        names.append(os.path.basename(calib_))

    R, T, K = np.array(R), np.array(T), np.array(K)
    K[:, 0, 0] = -K[:, 0, 0]
    K[:, 1, 1] = -K[:, 1, 1]

    return R, T, K, np.array(names)


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def obj_read_w_color(self, obj_path):
    vs = []
    fs = []
    cs = []
    with open(obj_path, 'r') as obj_f:
        lines = obj_f.readlines()
        for line in lines:
            line_ = line.split(' ')
            if len(line_) < 4:
                line_ = line.split('\t')
            if line_[0] == 'v':
                v1 = float(line_[1])
                v2 = float(line_[2])
                v3 = float(line_[3])
                vs.append([v1, v2, v3])
                c1 = float(line_[4])
                c2 = float(line_[5])
                c3 = float(line_[6])
                cs.append([c1, c2, c3])
            if line_[0] == 'f':
                f1 = int(line_[1].split('/')[0]) - 1
                f2 = int(line_[2].split('/')[0]) - 1
                f3 = int(line_[3].split('/')[0]) - 1
                fs.append([f1, f2, f3])
    vs = np.array(vs)
    fs = np.array(fs)
    cs = np.array(cs)

    return vs, fs, cs

def obj_read(obj_path):
    vs = []
    fs = []
    with open(obj_path, 'r') as obj_f:
        lines = obj_f.readlines()
        for line in lines:
            line_ = line.split(' ')
            if len(line_) < 4:
                line_ = line.split('\t')
            if line_[0] == 'v':
                v1 = float(line_[1])
                v2 = float(line_[2])
                v3 = float(line_[3])
                vs.append([v1, v2, v3])
            if line_[0] == 'f':
                f1 = int(line_[1].split('/')[0]) - 1
                f2 = int(line_[2].split('/')[0]) - 1
                f3 = int(line_[3].split('/')[0]) - 1
                fs.append([f1, f2, f3])
    vs = np.array(vs)
    fs = np.array(fs)

    return vs, fs