import numpy as np
from weather_repr.helpers import compute_wind_speed_field
import matplotlib.pyplot as plt

def flatten_pyramid(pyramid):
    return np.concatenate([p.flatten() for p in pyramid])

# def pool_grid(field, out_h, out_w, mode="mean"):
#     """
#     Downsample a 2D field to (out_h, out_w) using pooling.
#     mode: 'mean', 'max', 'min'
#     """
#     H, W = field.shape

#     stride_h = H // out_h
#     stride_w = W // out_w

#     pooled = np.zeros((out_h, out_w))

#     for i in range(out_h):
#         for j in range(out_w):
#             h_start = i * stride_h
#             h_end = (i + 1) * stride_h
#             w_start = j * stride_w
#             w_end = (j + 1) * stride_w

#             patch = field[h_start:h_end, w_start:w_end]

#             if mode == "mean":
#                 pooled[i, j] = np.mean(patch)
#             elif mode == "max":
#                 pooled[i, j] = np.max(patch)
#             elif mode == "min":
#                 pooled[i, j] = np.min(patch)
#             else:
#                 raise ValueError("Invalid pooling mode")

#     return pooled

# def multi_channel_pool(field, out_h, out_w):
#     mean_pool = pool_grid(field, out_h, out_w, mode="mean")
#     max_pool  = pool_grid(field, out_h, out_w, mode="max")
#     min_pool  = pool_grid(field, out_h, out_w, mode="min")

#     # Stack into channels → shape: (C, H, W)
#     return np.stack([mean_pool, max_pool, min_pool], axis=0)

# def build_weather_pyramid(field, scales=[20, 10, 5]):
#     """
#     Returns list of pooled grids at different scales
#     """
#     pyramid = []

#     for s in scales:
#         pooled = pool_grid(field, s, s, mode="mean")
#         pyramid.append(pooled)

#     return pyramid

# def build_multi_scale_multi_channel(field, scales=[20, 10, 5]):
#     """
#     Returns list of (C, H, W) tensors
#     """
#     pyramid = []

#     for s in scales:
#         pooled = multi_channel_pool(field, s, s)
#         pyramid.append(pooled)

#     return pyramid

def vector_pool_grid(u, v, out_h, out_w):
    """
    Pools vector field (u, v) into:
    - mean_u
    - mean_v
    - max_speed
    """
    H, W = u.shape

    stride_h = H // out_h
    stride_w = W // out_w

    mean_u = np.zeros((out_h, out_w))
    mean_v = np.zeros((out_h, out_w))
    max_speed = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            h_end = (i + 1) * stride_h
            w_start = j * stride_w
            w_end = (j + 1) * stride_w

            patch_u = u[h_start:h_end, w_start:w_end]
            patch_v = v[h_start:h_end, w_start:w_end]

            # Mean vector (preserves direction)
            mean_u[i, j] = np.mean(patch_u)
            mean_v[i, j] = np.mean(patch_v)

            # Max magnitude (captures storms/extremes)
            speed_patch = np.sqrt(patch_u**2 + patch_v**2)
            max_speed[i, j] = np.max(speed_patch)

    return mean_u, mean_v, max_speed

def multi_channel_vector_pool(u, v, out_h, out_w):
    mean_u, mean_v, max_speed = vector_pool_grid(u, v, out_h, out_w)

    # Stack channels: (C, H, W)
    return np.stack([mean_u, mean_v, max_speed], axis=0)

"""this will be the output now:

[
  mean_u,       # directional x-component
  mean_v,       # directional y-component
  max_speed     # extreme weather
]"""

def build_vector_pyramid(u, v, scales=[20, 10, 5]):
    pyramid = []

    for s in scales:
        pooled = multi_channel_vector_pool(u, v, s, s)
        pyramid.append(pooled)

    return pyramid

def get_global_weather_representation(world):
    u = world.wind_u
    v = world.wind_v

    pyramid = build_vector_pyramid(
        u, v,
        scales=[20, 10, 5]
    )

    return flatten_pyramid(pyramid)



