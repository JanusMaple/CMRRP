"""
Read task_plan_test Saved Data and Plot Motion Planning Results
"""
import os
import io
import sys
sys.path.append('..')
import tqdm
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from GMRC import GMRC
from AtlasRRT import *
from CMMRC import CMMRC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Planning Parameters and Print Settings """
Chart.device = device
Chart.epsilon = 0.05
Chart.rho = 0.1
Chart.beta = 2.0
AtlasRRTree.delta = 0.01
AtlasRRTree.lambda_ = 3.0
AtlasRRTree.gamma = 0.01
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="Select Parameters for Motion Plan Test")
parser.add_argument('--f', type=str, default=None,
                    help='Filename of the task-plan path data, excluding ".pt"')
parser.add_argument('--m', type=str, default="MCTS",
                    help='The task plan method that created the data')
default_dirname = os.path.dirname(os.path.abspath(__file__)) + "/data"
parser.add_argument('--dir', type=str, default=default_dirname,
                    help='Data directory for task plan results')
args = parser.parse_args()

filename = args.f
method = args.m
dirname = args.dir
save_dirname = dirname + "/motion_plan_results"
os.makedirs(save_dirname, exist_ok=True)

if filename is None:
    raise ValueError("Please Provide a Filename: x_yyyyyy")

data = torch.load(dirname + "/task_plan_results/" + filename + ".pt",
                  weights_only=False)
if method == "IMT_BFS":
    path = data[0][0]
elif method == "MCTS":
    path = data[1][0]
else:
    raise ValueError("Please Provide Valid Method Name: IMT_BFS/MCTS")

m = int(filename[:-7])
seed = int(filename[-6:])
if len(data) > 2:
    gmrc_1 = data[2]
    gmrc_2 = data[3]
else:
    gmrc_1 = GMRC.get_random_configuration(m=m, seed=seed)
    gmrc_2 = GMRC.get_random_configuration(m=m, seed=1000000+seed)

""" Find all gmrc pairs to be motion planned """
gmrc_pairs = []
for i in range(len(path) - 1):
    if not path[i].g_depth == path[i + 1].g_depth:
        gmrc_pairs.append((path[i].gmrc, path[i + 1].gmrc))
tar_gmrc = path[-1].gmrc.copy()
correspondence = path[-1].cgf_manager.correspondence
for i in range(tar_gmrc.m):
    j = correspondence[i * 2] // 2
    if correspondence[i * 2] % 2 == 0:
        s = 1
    else:
        s = -1
    tar_gmrc.bend_angs[i] = s * gmrc_2.bend_angs[j]
tar_gmrc.update_all_module_geometry()
tar_gmrc.update_all_module_collider()
gmrc_pairs.append((path[-1].gmrc, tar_gmrc))

print(f"Start motion planning for {len(gmrc_pairs)} subtasks")
mp_paths = []
for i, gmrc_pair in tqdm.tqdm(zip(range(len(gmrc_pairs)), gmrc_pairs)):
    cmmrc = CMMRC(gmrc_pair[0], gmrc_pair[1], device)
    xs = torch.tensor(gmrc_pair[0].bend_angs, 
                      dtype=torch.float, device = device, requires_grad=True)
    xg = torch.tensor(gmrc_pair[1].bend_angs, 
                      dtype=torch.float, device = device, requires_grad=True)
    planner = AtlasRRTPlanner(xs, xg, cmmrc.constraint_func, cmmrc.collision_func)
    mp_paths.append(planner.plan(1800, 20))

""" Plot as GIF and save it with path to disk """
xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
 
for gmrc_pair, mp_path in zip(gmrc_pairs, mp_paths):
    plot_gmrc = gmrc_pair[0].copy()
    for way_pnt in mp_path:
        plot_gmrc.bend_angs = way_pnt.detach().cpu().numpy()
        plot_gmrc.update_all_module_geometry()
        plot_gmrc.update_all_module_collider()
 
        # harvest extents from geometries and colliders
        for i in range(plot_gmrc.m):
            # module_geometries[i] assumed iterable of arrays of shape (N,2) or similar
            for geom in plot_gmrc.module_geometries[i]:
                arr = np.asarray(geom)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    xmin = min(xmin, np.min(arr[:,0])); xmax = max(xmax, np.max(arr[:,0]))
                    ymin = min(ymin, np.min(arr[:,1])); ymax = max(ymax, np.max(arr[:,1]))
 
            # colliders (each 'line' has .xy)
            for line in plot_gmrc.module_colliders[i][1].geoms:
                x, y = line.xy
                xmin = min(xmin, np.min(x)); xmax = max(xmax, np.max(x))
                ymin = min(ymin, np.min(y)); ymax = max(ymax, np.max(y))
 
# add a small padding so strokes/labels don't touch the edge
dx, dy = (xmax - xmin), (ymax - ymin)
pad = 0.05 * max(dx, dy) if max(dx, dy) > 0 else 1.0
X_LIM = (xmin - pad, xmax + pad)
Y_LIM = (ymin - pad, ymax + pad)
 
print("Motion Plan Completed; Save GIF to disk...")
fig, gif_ax = plt.subplots()
gif_frames = []
for gmrc_pair, mp_path in zip(gmrc_pairs, mp_paths):
    plot_gmrc = gmrc_pair[0].copy()
    for way_pnt in mp_path:
        plot_gmrc.bend_angs = way_pnt.detach().cpu().numpy()
        plot_gmrc.update_all_module_geometry()
        plot_gmrc.update_all_module_collider()

        gif_ax.clear()
        gif_ax.set_aspect('equal')
        gif_ax.set_xlim(*X_LIM)
        gif_ax.set_ylim(*Y_LIM)
        gif_ax.axis('off')
        leaf_count = plot_gmrc.w + plot_gmrc.v
        for i in range(plot_gmrc.m):
            g1n = f"H{plot_gmrc.module2gripper[0][i]}"
            mn = f"{i}"
            g2n = f"T{plot_gmrc.module2gripper[1][i]}"
            h_node = None
            t_node = None
            if plot_gmrc.module2gripper[0][i] % 3 == 0:
                h_node = plot_gmrc.module2gripper[0][i] // 3
            elif plot_gmrc.module2gripper[0][i] < 0:
                h_node = leaf_count
                leaf_count = leaf_count + 1
            if plot_gmrc.module2gripper[1][i] % 3 == 0:
                t_node = plot_gmrc.module2gripper[1][i] // 3
            elif plot_gmrc.module2gripper[1][i] < 0:
                t_node = leaf_count
                leaf_count = leaf_count + 1
            GMRC.draw_module(
                gif_ax, 
                plot_gmrc.module_geometries[i][0], 
                plot_gmrc.module_geometries[i][1], 
                plot_gmrc.module_geometries[i][2], 
                g1n, 
                mn, 
                g2n,
                True,
                h_node,
                t_node
            )
            for line in plot_gmrc.module_colliders[i][1].geoms:
                x, y = line.xy
                gif_ax.plot(x, y, color = 'dimgray')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        gif_frames.append(Image.open(buf))

duration = 20
for _ in range(1000 // duration):
    gif_frames.append(gif_frames[-1])

gif_frames[0].save(
    save_dirname + "/" + filename + "_" + method + ".gif",
    format='GIF',
    save_all=True,
    append_images=gif_frames[1:],
    duration=duration,          # Duration between frames in milliseconds
    loop=0                      # Loop indefinitely
)
