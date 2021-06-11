from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF

class KERNEL_MODE(Enum):
    FULL: int = 0
    CROSS: int = 1
    DIAMOND: int = 2

class BLUR_TYPE(Enum):
    BILATERAL_FILTER: int = 0
    GAUSSIAN_FILTER: int = 1

class IP_Basic(nn.Module):
    def __init__(self,
        max_depth: float = 100.0,
        kernel_size: int = 5,
        # kernel_mode: int = KERNEL_MODE.DIAMOND,
        extrapolate: bool = False,
        blur_mode: int = BLUR_TYPE.GAUSSIAN_FILTER,
        invert: bool = True,
        min_depth: float = 0.1,
    ):
        super(IP_Basic, self).__init__()

        self.max_depth: float = max_depth
        self.custom_kernel_size: int = kernel_size
        # self.custom_kernel: torch.Tensor = self._get_kernel(self.custom_kernel_size, kernel_mode)
        self.extrapolate: bool = extrapolate
        self.blur_mode: int = blur_mode
        self.invert: bool = invert
        self.min_depth: float = min_depth

    def _get_kernel(self,
        kernel_size: int,
        mode: int = KERNEL_MODE.FULL
    ) -> torch.Tensor:
        assert kernel_size % 2 == 1, '"kernel_size" must be odd.'

        if mode == KERNEL_MODE.FULL:
            return torch.ones(kernel_size, kernel_size)
        elif mode == KERNEL_MODE.CROSS:
            kernel = torch.zeros(kernel_size, kernel_size)
            center : int = kernel_size // 2
            kernel[center, :] = 1
            kernel[:, center] = 1
            return kernel
        elif mode == KERNEL_MODE.DIAMOND:
            kernel = torch.zeros(kernel_size, kernel_size)
            center : int = kernel_size // 2
            half_end : int = center + 1

            for col_start in range(half_end):
                row_start = half_end - col_start - 1
                row_end = kernel_size - row_start
                col_end = kernel_size - col_start
                step = max(1, col_start * 2)
                kernel[row_start:row_end:step, col_start:col_end] = 1
            return kernel
        else:
            raise NotImplementedError(f'Select "mode" from {[v.name for v in KERNEL_MODE]}')

    def dilation(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)

    def erosion(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return -self.dilation(-x, kernel_size)

    def closing(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        y = self.dilation(x, kernel_size)
        return self.erosion(y, kernel_size)

    def opening(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        y = self.erosion(x, kernel_size)
        return self.dilation(y, kernel_size)

    def median_blur(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding: int = kernel_size // 2

        window_range: int = kernel_size * kernel_size
        kernel: torch.Tensor = torch.eye(window_range, window_range, dtype=x.dtype, device=x.device).view(window_range, 1, kernel_size, kernel_size)
        b, c, h, w = x.shape

        features: torch.Tensor = F.conv2d(x.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1).view(b, c, -1, h, w)

        return torch.median(features, dim=2).values

    def forward(self, in_depth: torch.Tensor) -> torch.Tensor:
        # Invert
        if self.invert is True:
            depth_map: torch.Tensor = torch.where(in_depth > self.min_depth, self.max_depth - in_depth, in_depth)
        else:
            depth_map: torch.Tensor = in_depth

        # Dilate
        depth_map = self.dilation(depth_map, self.custom_kernel_size)

        # Hole closing
        depth_map = self.closing(depth_map, 5)

        # Fill empty spaces with dilated values
        empty_pixels = (depth_map < self.min_depth)
        dilated = self.dilation(depth_map, 7)
        depth_map[empty_pixels] = dilated[empty_pixels]

        # Extend highest pixel to top of image
        if self.extrapolate is True:
            raise NotImplementedError()

        # Median blur
        depth_map = self.median_blur(depth_map, 5)

        # Bilateral or Gaussian blur
        if self.blur_mode == BLUR_TYPE.BILATERAL_FILTER:
            raise NotImplementedError()
        elif self.blur_mode == BLUR_TYPE.GAUSSIAN_FILTER:
            blurred = vF.gaussian_blur(depth_map, 5)
            depth_map = torch.where(depth_map > self.min_depth, blurred, depth_map)

        # Invert
        if self.invert is True:
            depth_map: torch.Tensor = torch.where(depth_map > self.min_depth, self.max_depth - depth_map, depth_map)
        else:
            depth_map: torch.Tensor = depth_map

        return depth_map
