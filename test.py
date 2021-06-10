import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from ip_basic import IP_Basic

from h5dataloader.pytorch import HDF5Dataset
from pointsmap import depth2colormap
import cv2

ds = HDF5Dataset(['/workspace/dataset/kitti360_seq03.hdf5'], '/workspace/dataset/kitti360-5class-80m.json', True)
dl = DataLoader(ds, batch_size=1)

ipb = IP_Basic(max_depth=80.0)

for batch in dl:
    depth_tensor: torch.Tensor = batch['depth']
    z = torch.zeros_like(depth_tensor)
    depth_tensor = torch.where(depth_tensor > 80.0, z, depth_tensor)

    x_cv = depth_tensor.numpy()[0]
    x_cv = np.transpose(x_cv, [1, 2, 0]).squeeze()
    colormap = depth2colormap(x_cv, min=0.0, max=100.0, invert=True)
    cv2.imwrite('in.png', colormap)

    y: torch.Tensor = ipb(depth_tensor)

    y_cv: np.ndarray = y.numpy()[0]
    y_cv = np.transpose(y_cv, [1, 2, 0]).squeeze()
    colormap = depth2colormap(y_cv, min=0.0, max=100.0, invert=True)
    print(y_cv)
    print(y_cv.shape)
    cv2.imwrite('out.png', colormap)
    break
