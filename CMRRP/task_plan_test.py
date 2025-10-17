"""
Compare Plan Speed Between IMT_BFS and MCTS
"""

import sys
sys.path.append('..')
sys.path.append('../GENN')
sys.path.append('../GGNN')
import os
import re
import time
import argparse
from pathlib import Path
from GMRC import GMRC
from CMRRP import *
import torch
from GENN import GENN
from GENN import DegreeEmbedding as GEDE
from GENN import SequentialPooling as GESP
from GGNN import GGNN
from GGNN import DegreeEmbedding as GGDE
from GGNN import SequentialPooling as GGSP

def format_hms(seconds, *, trim_leading_zero=True, decimals=1):
    """Return e.g. 3671 -> '1h 1m 11s', 65 -> '1m 5s'."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)

    h, rem = divmod(int(seconds), 3600)
    m, s_int = divmod(rem, 60)

    frac = seconds - int(seconds)
    if decimals > 0:
        s = s_int + frac
        s_str = f"{s:.{decimals}f}".rstrip("0").rstrip(".")
    else:
        s = round(s_int + frac)
        if s == 60:
            s = 0; m += 1
        if m == 60:
            m = 0; h += 1
        s_str = str(int(s))

    parts = []
    if not trim_leading_zero or h:
        parts.append(f"{h}h")
    if not trim_leading_zero or h or m:
        parts.append(f"{m}m")
    parts.append(f"{s_str}s")
    return sign + " ".join(parts)

def iter_m_data(dir_name, m):
    pat = re.compile(rf"^{re.escape(str(m))}_(\d{{6}})\.pt$")
    paths = []
    for p in Path(dir_name).iterdir():
        m = pat.match(p.name)
        if m:
            paths.append((int(m.group(1)), p))
    for seed, p in sorted(paths):
        yield seed, torch.load(p, weights_only=False)

def plan_and_time(cmrrp: CMRRP, gmrc_1: GMRC, gmrc_2: GMRC, method: str, time_budget: float):
    start_time = time.time()
    path = cmrrp.plan(gmrc_1, gmrc_2, method, time_budget, False)
    end_time = time.time()
    if path is not None:
        path = [node.strip2path() for node in path]
        g_dis = path[-1].g_depth
        delta_time = end_time - start_time
    else:
        g_dis = None
        delta_time = None
    return (path, g_dis, delta_time)

def print_single_info(bfs_path, bfs_dis, bfs_time, mcts_path, mcts_dis, mcts_time):
    if bfs_path is None:
        print("IMT_BFS \033[91mFailed\033[0m; ", end="")
    else:
        if mcts_dis is None or bfs_dis <= mcts_dis:
            print(f"IMT_BFS finds \033[92m{bfs_dis}\033[0m", end="")
        else:
            print(f"IMT_BFS finds \033[91m{bfs_dis}\033[0m", end="")
        if mcts_time is None or bfs_time <= mcts_time:
            print(f"-path in \033[92m{format_hms(bfs_time)}\033[0m; ", end="")
        else:
            print(f"-path in \033[91m{format_hms(bfs_time)}\033[0m; ", end="")
    
    if mcts_path is None:
        print("MCTS \033[91mFailed\033[0m")
    else:
        if bfs_dis is None or mcts_dis <= bfs_dis:
            print(f"MCTS finds \033[92m{mcts_dis}\033[0m", end="")
        else:
            print(f"MCTS finds \033[91m{mcts_dis}\033[0m", end="")
        if bfs_time is None or mcts_time <= bfs_time:
            print(f"-path in \033[92m{format_hms(mcts_time)}\033[0m")
        else:
            print(f"-path in \033[91m{format_hms(mcts_time)}\033[0m")

def print_comprehensive_info(bfs_times, bfs_distances, mcts_times, mcts_distances):
    bfs_succ_times = [dt for dt in bfs_times if dt is not None]
    bfs_mean = np.mean(np.array(bfs_succ_times))
    bfs_std = np.std(np.array(bfs_succ_times))
    bfs_succ_rate = len(bfs_succ_times) / len(bfs_distances)
    print(f"IMT_BFS takes \033[96m{format_hms(bfs_mean)}±{format_hms(bfs_std)}\033[0m", end="")
    print(f" with Success Rate: \033[96m{bfs_succ_rate * 100:.1f}%\033[0m")
    mcts_succ_times = [dt for dt in mcts_times if dt is not None]
    mcts_mean = np.mean(np.array(mcts_succ_times))
    mcts_std = np.std(np.array(mcts_succ_times))
    mcts_succ_rate = len(mcts_succ_times) / len(mcts_distances)
    optimal_times = 0
    both_solved_times = 0
    for bfs_dis, mcts_dis in zip(bfs_distances, mcts_distances):
        if bfs_dis is not None and mcts_dis is not None:
            both_solved_times = both_solved_times + 1
            if mcts_dis <= bfs_dis:
                optimal_times = optimal_times + 1

    print(f"MCTS takes \033[96m{format_hms(mcts_mean)}±{format_hms(mcts_std)}\033[0m", end="")
    print(f" with Success Rate: \033[96m{mcts_succ_rate * 100:.1f}%\033[0m", end="")
    print(f"; Optimal Rate: \033[96m{optimal_times / both_solved_times * 100:.1f}%\033[0m")

def main():
    parser = argparse.ArgumentParser(
        description="Select Parameters for Task Plan Test")
    parser.add_argument('--m', type=int, default=3,
                        help='Number of Modules')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of Tests')
    parser.add_argument('--t', type=float, default=60.0,
                        help='Time Budget (s)')
    parser.add_argument('--s', type=int, default=100000,
                        help='Seed for the first round')
    parser.add_argument("--generate_mode", action=argparse.BooleanOptionalAction,
                    default=True, help="Geberate Data and Store Them to Disk")
    parser.add_argument("--analyze_mode", action=argparse.BooleanOptionalAction,
                    default=False, help="Analyze Data by Loading From Disk")
    default_dirname = os.path.dirname(os.path.abspath(__file__)) + "/data"
    parser.add_argument('--dir', type=str, default=default_dirname,
                        help='Where to save planned results')
    args = parser.parse_args()

    m = args.m
    n = args.n
    time_budget = args.t
    dir_name = args.dir + "/task_plan_results"
    os.makedirs(dir_name, exist_ok=True)
    seed = args.s

    if n > 1000:
        raise ValueError("More than 1000 tests will take a loooooong time. ")
    if args.analyze_mode:
        print("\033[36mAnalyze Mode\033[0m \033[92mOn\033[0m")
        bfs_times = []
        mcts_times = []
        bfs_distances = []
        mcts_distances = []
        for seed, data in iter_m_data(dir_name, m):
            print(f"Seed: {seed}; ", end="")
            bfs_path, bfs_dis, bfs_time = data[0]
            mcts_path, mcts_dis, mcts_time = data[1]
            gmrc_1 = data[2]
            gmrc_2 = data[3]
            bfs_times.append(bfs_time)
            bfs_distances.append(bfs_dis)
            mcts_times.append(mcts_time)
            mcts_distances.append(mcts_dis)
            print_single_info(bfs_path, bfs_dis, bfs_time, mcts_path, mcts_dis, mcts_time)
        if len(bfs_distances) > 0:
            print_comprehensive_info(bfs_times, bfs_distances, mcts_times, mcts_distances)
        else:
            print("\033[91mNo Data Found ...\033[0m")
        return
    elif args.generate_mode:
        print("\033[36mGenerate Mode\033[0m \033[92mOn\033[0m")
    else:
        raise ValueError("Please Select \033[36mgenerate_mode\033[0m \033[91mOR\033[0m \033[36manalyze_mode\033[0m")
    
    device = torch.device('cpu')

    genn_degree_embedding = GEDE(embed_dim=16, device=device)
    genn = GENN(16, 32, 32, device)
    genn_pooling = GESP(32, 32, 16, device)
    checkpoint = torch.load("../GENN/model/bfs_trained_model.pth", map_location=device)
    genn_degree_embedding.load_state_dict(checkpoint['degree_embedding'])
    genn.load_state_dict(checkpoint['gnn'])
    genn_pooling.load_state_dict(checkpoint['pooling'])
    genn_degree_embedding.eval()
    genn.eval()
    genn_pooling.eval()

    ggnn_degree_embedding = GGDE(embed_dim=2, device=device)
    ggnn = GGNN(2, 2, 2, device)
    ggnn_pooling = GGSP(2, 2, 1, device)
    checkpoint = torch.load("../GGNN/model/ggnn_hash_model.pth", map_location=device)
    ggnn_degree_embedding.load_state_dict(checkpoint['degree_embedding'])
    ggnn.load_state_dict(checkpoint['gnn'])
    ggnn_pooling.load_state_dict(checkpoint['pooling'])
    ggnn_degree_embedding.eval()
    ggnn.eval()
    ggnn_pooling.eval()

    GMRC.suppress_spawn_err = True
    cmrrp = CMRRP(
        ggnn, ggnn_degree_embedding, ggnn_pooling,
        genn, genn_degree_embedding, genn_pooling,
        device
    )

    """
    Warm up MCTS by creating all workders for ParOptimizer
    """
    gmrc_1 = GMRC.get_random_configuration(m=3, seed=726130)
    gmrc_2 = GMRC.get_random_configuration(m=3, seed=1726130)
    _ = cmrrp.plan(gmrc_1, gmrc_2, "MCTS")

    """
    Test for n times and record the results
    """
    num_tests = 0
    bfs_times = []
    mcts_times = []
    bfs_distances = []
    mcts_distances = []
    print(f"\033[94mm = {m}; Start Testing for {n} Rounds: \033[0m")
    while num_tests < n:
        gmrc_1 = GMRC.get_random_configuration(m=m, seed=seed)
        gmrc_2 = GMRC.get_random_configuration(m=m, seed=1000000+seed)
        if gmrc_1.successfully_spawned and gmrc_2.successfully_spawned:
            print(f"Round-{num_tests}; Seed: {seed}; ", end="", flush=True)

            bfs_path, bfs_dis, bfs_time = plan_and_time(
                cmrrp, gmrc_1.copy(), gmrc_2.copy(), "IMT_BFS", time_budget)
            if CMRRP.finish_mode == 1:          # If No Solution Exists
                seed = seed + 1
                print("\033[33mSkipped: No Solution\033[0m")
                continue
            bfs_distances.append(bfs_dis)
            bfs_times.append(bfs_time)

            mcts_path, mcts_dis, mcts_time = plan_and_time(
                cmrrp, gmrc_1.copy(), gmrc_2.copy(), "MCTS", time_budget)
            mcts_distances.append(mcts_dis)
            mcts_times.append(mcts_time)

            data = ((bfs_path, bfs_dis, bfs_time),
                    (mcts_path, mcts_dis, mcts_time),
                    gmrc_1, gmrc_2)
            file_name = f"/{m}_{seed}.pt"
            torch.save(data, dir_name + file_name)

            print_single_info(bfs_path, bfs_dis, bfs_time, mcts_path, mcts_dis, mcts_time)
            num_tests = num_tests + 1
            seed = seed + 1
        else:
            seed = seed + 1
    print_comprehensive_info(bfs_times, bfs_distances, mcts_times, mcts_distances)
    ParOptimizer.cleanup_all_pools()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()