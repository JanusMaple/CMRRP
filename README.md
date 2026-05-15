# CMRRP
 
Code for **Continuum Modular Robot Reconfiguration Planning**.
 
This repository provides the implementation of the self-reconfiguration planning framework for Modular Self-Reconfigurable Continuum Robots (MSRCRs). The codebase includes representations of MSRCR topology, morphology, and shape, together with motion-planning and task-planning algorithms.
 
## Installation

```bash
git clone https://github.com/JanusMaple/CMRRP.git
cd CMRRP
conda env create -f environment.yml
conda activate cmrrp
cd CMRRP
unzip data.zip
cd ..
```
 
## Usage
 
### 1. Procedural Configuration Generation
 
A bottom-up procedural configuration generation (PCG) pipeline is implemented as follows:
 
**Step 1. Topology generation in TMRC**  
The configuration model is used to generate a random mixed graph representing the topology of an MSRCR.
 
**Step 2. Rotation-system generation in EMRC**  
Random directions are assigned to all cycles in the MSRCR to construct a random rotation system.
 
**Step 3. Morphology and Geometry generation in GMRC**  
Bending angles and grasping angles are randomly generated and assigned to the MSRCR while enforcing connectivity and self-collision constraints. A finite number of attempts is allowed; if no feasible set of bending and grasping angles is found, the generation process reports failure.
 
```python
from GMRC import GMRC
import matplotlib.pyplot as plt
 
module_number = 5           # Number of Modules in MSRCR
 
gmrc = GMRC.get_random_configuration(m=module_number)
 
gmrc.show_geometry()        # Draw Sketch of MSRCR Shape
plt.show(block=True)
 
gmrc.print_all()            # Print All Information
```
 
### 2. Task and Motion Planning for Self-Reconfiguration
 
A unified task and motion planning pipeline for MSRCR self-reconfiguration planning:
 
**Motion Planning**
Atlas-RRT* is used for motion planning on the constraint manifold.
 
**Task Planning**
HEART-MCTS is implemented for task planning of self-reconfiguration.
 
```python
''' ------ Procedural Generation ------ '''

from GMRC import GMRC
module_number = 5
GMRC.suppress_spawn_err = True
for i in range(1000):
    gmrc_1 = GMRC.get_random_configuration(m=module_number)
    if gmrc_1.successfully_spawned:
        break
for i in range(1000):
    gmrc_2 = GMRC.get_random_configuration(m=module_number)
    if gmrc_2.successfully_spawned:
        break

''' ------ GAS-GNN Preparation ------ '''

import sys
import torch
sys.path.append('GENN')
sys.path.append('GGNN')
from GENN import GENN
from GENN import DegreeEmbedding as GEDE
from GENN import SequentialPooling as GESP
from GGNN import GGNN
from GGNN import DegreeEmbedding as GGDE
from GGNN import SequentialPooling as GGSP

device = torch.device('cpu')

genn_degree_embedding = GEDE(embed_dim=16, device=device)
genn = GENN(16, 32, 32, device)
genn_pooling = GESP(32, 32, 16, device)
checkpoint = torch.load("GENN/model/bfs_trained_model.pth", map_location=device)
genn_degree_embedding.load_state_dict(checkpoint['degree_embedding'])
genn.load_state_dict(checkpoint['gnn'])
genn_pooling.load_state_dict(checkpoint['pooling'])
genn_degree_embedding.eval()
genn.eval()
genn_pooling.eval()

ggnn_degree_embedding = GGDE(embed_dim=2, device=device)
ggnn = GGNN(2, 2, 2, device)
ggnn_pooling = GGSP(2, 2, 1, device)
checkpoint = torch.load("GGNN/model/ggnn_hash_model.pth", map_location=device)
ggnn_degree_embedding.load_state_dict(checkpoint['degree_embedding'])
ggnn.load_state_dict(checkpoint['gnn'])
ggnn_pooling.load_state_dict(checkpoint['pooling'])
ggnn_degree_embedding.eval()
ggnn.eval()
ggnn_pooling.eval()

''' ------ Task Planning ------ '''

sys.path.append("CMRRP")
from CMRRP import *

cmrrp = CMRRP(
    ggnn, ggnn_degree_embedding, ggnn_pooling,
    genn, genn_degree_embedding, genn_pooling,
    device
)
method = "MCTS"     # MCTS, BFS or IAB_BFS
path = cmrrp.plan(gmrc_1, gmrc_2, method, 3600 * 24)
ParOptimizer.cleanup_all_pools()

''' ------ Motion Planning ------ '''

import tqdm
from AtlasRRT import *
from CMMRC import CMMRC

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Chart.device = device
Chart.epsilon = 0.05
Chart.rho = 0.1
Chart.beta = 2.0
AtlasRRTree.delta = 0.01
AtlasRRTree.lambda_ = 3.0
AtlasRRTree.gamma = 0.01

mp_paths = []
for i, gmrc_pair in tqdm.tqdm(zip(range(len(gmrc_pairs)), gmrc_pairs)):
    cmmrc = CMMRC(gmrc_pair[0], gmrc_pair[1], device)
    xs = torch.tensor(gmrc_pair[0].bend_angs, 
                      dtype=torch.float, device = device, requires_grad=True)
    xg = torch.tensor(gmrc_pair[1].bend_angs, 
                      dtype=torch.float, device = device, requires_grad=True)
    planner = AtlasRRTPlanner(xs, xg, cmmrc.constraint_func, cmmrc.collision_func)
    mp_paths.append(planner.plan(120, 20))

""" Generate GIF Save to Disk """
import io
from PIL import Image
import matplotlib.pyplot as plt

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

filename = f"random_test_{method}"
gif_frames[0].save(
    filename + ".gif",
    format='GIF',
    save_all=True,
    append_images=gif_frames[1:],
    duration=duration,          # Duration between frames in milliseconds
    loop=0                      # Loop indefinitely
)
```
### 3. Benchmark Task Planning Methods
Randomly generate 100 pairs of MSRCR configurations and perform task planning for self-reconfiguration between each pair. Example usage is shown below. If `data.zip` is extracted to `CMRRP/CMRRP/data/`, the analysis mode can be used directly.

```bash
cd CMRRP
mkdir data
python task_plan_test.py --m 3 --n 100 --t 60.0 --s 100000
python task_plan_test.py --m 3 --analyze_mode
python task_plan_test.py --m 4 --n 100 --t 600.0 --s 100000
python task_plan_test.py --m 5 --n 100 --t 100000.0 --s 100000
python task_plan_test.py --m 6 --analyze_mode
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
