import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

import run_nerf
import run_nerf_helpers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ### Load trained network weights
# Run `bash download_example_weights.sh` in the root directory if you need to download the Lego example weights

# In[ ]:


basedir = './logs'
expname = 'simple_cube_transparent'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())

parser = run_nerf.config_parser()
ft_str = ''
ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, '300000.tar'))
args = parser.parse_args('--config {} '.format(config) + ft_str)

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)


NeRF = render_kwargs_test['network_fine']
# N, chunk = 256, 1024 * 64
N, chunk = 64, 1024 * 64

t = np.linspace(-6, 6, N + 1)
query_points = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
print(query_points.shape)
flat = torch.from_numpy(query_points.reshape([-1, 3])).to(device)

query_fn = render_kwargs_test['network_query_fn']

sigma = []
for i in range(0, flat.shape[0], chunk):
    pts = flat[i:i + chunk, None, :]
    viwedirs = torch.zeros_like(flat[i:i + chunk])
    raw = query_fn(pts, viwedirs, NeRF)
    sigma.append(raw[..., -1])
# density = torch.concat(sigma, dim=0)
density = torch.concat(sigma, dim=0).detach().cpu().numpy().squeeze()
plt.hist(np.maximum(0, density), log=True)
plt.savefig('density.png')

import mcubes
import trimesh

threshold = 5

# vertices, triangles = mcubes.marching_cubes(density.reshape(257, 257, -1), threshold)
vertices, triangles = mcubes.marching_cubes(density.reshape(65, 65, -1), threshold)
print('done', vertices.shape, triangles.shape)
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export('000.ply')