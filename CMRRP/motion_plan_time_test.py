"""
Batch motion-planning timing benchmark.

Supports two input sources:
  1. data/task_plan_results/*.pt
     - extracts motion subtasks from saved task-planning paths.
  2. data/random_grasp_pairs/*.pt
     - runs motion planning directly on randomly generated single-grasp pairs.

By default, timing is feasibility-only: the script stops at the first feasible
Atlas-RRT path and does not include the Atlas-RRT* refinement/optimization phase.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import tqdm

sys.path.append('..')

from GMRC import GMRC
from AtlasRRT import Chart, AtlasRRTree, AtlasRRTPlanner
from CMMRC import CMMRC


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Match the defaults used in motion_plan_test.py
Chart.device = device
Chart.epsilon = 0.05
Chart.rho = 0.1
Chart.beta = 2.0
AtlasRRTree.delta = 0.01
AtlasRRTree.lambda_ = 3.0
AtlasRRTree.gamma = 0.01

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=3)

InputSource = Literal['task_plan_results', 'random_grasp_pairs']


def format_hms(seconds: float, *, trim_leading_zero: bool = True, decimals: int = 1) -> str:
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
            s = 0
            m += 1
        if m == 60:
            m = 0
            h += 1
        s_str = str(int(s))

    parts: list[str] = []
    if not trim_leading_zero or h:
        parts.append(f"{h}h")
    if not trim_leading_zero or h or m:
        parts.append(f"{m}m")
    parts.append(f"{s_str}s")
    return sign + " ".join(parts)


def iter_saved_cases(input_dir: str, m: int | None = None) -> Iterable[tuple[str, int, int, object]]:
    """Iterate over saved ``{m}_{seed}.pt`` files from either supported source."""
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if m is None:
        pat = re.compile(r"^(\d+)_(\d+)\.pt$")
    else:
        pat = re.compile(rf"^{re.escape(str(m))}_(\d+)\.pt$")

    matches: list[tuple[int, int, str, Path]] = []
    for p in path.iterdir():
        hit = pat.match(p.name)
        if not hit:
            continue
        if m is None:
            cur_m = int(hit.group(1))
            seed = int(hit.group(2))
        else:
            cur_m = m
            seed = int(hit.group(1))
        matches.append((cur_m, seed, p.stem, p))

    matches.sort(key=lambda x: (x[0], x[1], x[2]))
    for cur_m, seed, stem, p in matches:
        yield stem, cur_m, seed, torch.load(p, weights_only=False)


def get_task_path(data: object, method: str):
    method = method.upper()
    if method == "IAB_BFS":
        return data[0][0]
    if method == "MCTS":
        return data[1][0]
    raise ValueError("Please provide a valid method name: IAB_BFS/MCTS")


def get_gmrc_pair_from_task_data(data: object, m: int, seed: int) -> tuple[GMRC, GMRC]:
    if len(data) > 2:
        return data[2], data[3]
    gmrc_1 = GMRC.get_random_configuration(m=m, seed=seed)
    gmrc_2 = GMRC.get_random_configuration(m=m, seed=1000000 + seed)
    return gmrc_1, gmrc_2


def build_motion_subtasks_from_task_path(path, gmrc_2: GMRC) -> list[tuple[GMRC, GMRC]]:
    """Extract grasp/release/final-alignment motion subtasks from a task path."""
    if path is None or len(path) == 0:
        return []

    gmrc_pairs: list[tuple[GMRC, GMRC]] = []

    for i in range(len(path) - 1):
        if path[i].g_depth != path[i + 1].g_depth:
            gmrc_pairs.append((path[i].gmrc, path[i + 1].gmrc))

    tar_gmrc = path[-1].gmrc.copy()
    correspondence = path[-1].cgf_manager.correspondence
    for i in range(tar_gmrc.m):
        j = correspondence[i * 2] // 2
        s = 1 if correspondence[i * 2] % 2 == 0 else -1
        tar_gmrc.bend_angs[i] = s * gmrc_2.bend_angs[j]
    tar_gmrc.update_all_module_geometry()
    tar_gmrc.update_all_module_collider()
    gmrc_pairs.append((path[-1].gmrc, tar_gmrc))

    return gmrc_pairs


def build_motion_subtasks_from_random_pair(data: dict) -> list[tuple[GMRC, GMRC]]:
    """Build the single motion subtask stored by random_grasp_pair_test.py."""
    if not isinstance(data, dict):
        raise TypeError(
            "random_grasp_pairs input is expected to be a dict with keys 'gmrc_1' and 'gmrc_2'."
        )
    if 'gmrc_1' not in data or 'gmrc_2' not in data:
        raise KeyError("random_grasp_pairs file must contain 'gmrc_1' and 'gmrc_2'.")
    return [(data['gmrc_1'], data['gmrc_2'])]


def build_motion_subtasks(
    data: object,
    *,
    source: InputSource,
    method: str,
    m: int,
    seed: int,
) -> list[tuple[GMRC, GMRC]]:
    if source == 'task_plan_results':
        path = get_task_path(data, method)
        _, gmrc_2 = get_gmrc_pair_from_task_data(data, m, seed)
        return build_motion_subtasks_from_task_path(path, gmrc_2)

    if source == 'random_grasp_pairs':
        return build_motion_subtasks_from_random_pair(data)  # type: ignore[arg-type]

    raise ValueError(f"Unsupported source: {source}")


def _plan_first_feasible_path(planner: AtlasRRTPlanner, total_budget: float) -> tuple[list[torch.Tensor], str | None]:
    """Return immediately when the first feasible Atlas-RRT connection is found.

    Returns a non-empty path and ``None`` on success. On failure, returns an
    empty path and a short failure reason.
    """
    trees = [planner.Ts, planner.Tg]
    i = 0
    start_time = time.time()

    while True:
        if time.time() - start_time > total_budget:
            print("\033[91mRunning out of time budget!\033[0m")
            return [], "Time budget exceeded"

        chart, ur = trees[i].sample(False)
        xr = chart.phi(ur)

        node_index_0 = trees[i].get_nearest_node(xr, chart)
        ni0 = trees[i].extend(xr, node_index_0, is_explore=True)
        xl0 = trees[i].nodes[ni0].x

        node_index_1 = trees[1 - i].get_nearest_node(xr)
        ni1 = trees[1 - i].extend(xr, node_index_1, is_explore=False)
        xl1 = trees[1 - i].nodes[ni1].x

        if torch.norm(xl0 - xl1, 2) < AtlasRRTree.delta:
            path_1st_half = trees[i].get_path(ni0)
            path_2nd_half = trees[1 - i].get_path(ni1)
            path_2nd_half.reverse()
            path = path_1st_half + path_2nd_half
            if i == 1:
                path.reverse()
            return path, None

        i = 1 - i


def plan_motion_subtask(
    gmrc_src: GMRC,
    gmrc_tgt: GMRC,
    *,
    total_budget: float,
    optim_budget: float,
    feasibility_only: bool = True,
) -> tuple[list[torch.Tensor], float, bool, str | None]:
    """Run one motion-planning subtask.

    Runtime errors raised inside Atlas-RRT, such as ``Exponential Map Time Out``,
    are treated as failed motion-planning attempts instead of aborting the full
    benchmark.
    """
    t0 = time.time()
    try:
        cmmrc = CMMRC(gmrc_src, gmrc_tgt, device)
        xs = torch.tensor(
            gmrc_src.bend_angs,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        xg = torch.tensor(
            gmrc_tgt.bend_angs,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        planner = AtlasRRTPlanner(xs, xg, cmmrc.constraint_func, cmmrc.collision_func)

        if feasibility_only:
            mp_path, failure_reason = _plan_first_feasible_path(planner, total_budget)
        else:
            mp_path = planner.plan(total_budget, optim_budget)
            failure_reason = None if len(mp_path) > 0 else "No feasible path found within budget"

        dt = time.time() - t0
        success = len(mp_path) > 0
        if success:
            return mp_path, dt, True, None
        return mp_path, dt, False, failure_reason or "No feasible path found within budget"

    except RuntimeError as e:
        dt = time.time() - t0
        msg = str(e)
        if "Exponential Map Time Out" in msg:
            reason = "Exponential Map Time Out"
        else:
            reason = f"RuntimeError: {msg}"
        return [], dt, False, reason
    except Exception as e:
        dt = time.time() - t0
        return [], dt, False, f"{type(e).__name__}: {e}"


def summarize_results(results: list[dict]) -> None:
    if not results:
        print("\033[91mNo motion-planning timing results found.\033[0m")
        return

    case_total_times = [r["case_total_wall_time"] for r in results]
    case_total_success_times = [
        r.get(
            "case_total_wall_time_success_only",
            sum(
                t
                for t, ok in zip(r.get("subtask_wall_times", []), r.get("subtask_success", []))
                if ok
            ),
        )
        for r in results
    ]
    case_mean_times = [
        r["case_mean_subtask_wall_time"]
        for r in results
        if r["case_mean_subtask_wall_time"] is not None
    ]
    case_mean_success_times = [
        r["case_mean_subtask_wall_time_success_only"]
        for r in results
        if r.get("case_mean_subtask_wall_time_success_only") is not None
    ]
    all_subtask_times = [t for r in results for t in r["subtask_wall_times"]]
    successful_subtask_times = [
        t
        for r in results
        for t, ok in zip(r["subtask_wall_times"], r["subtask_success"])
        if ok
    ]

    total_cases = len(results)
    total_subtasks = sum(r["num_subtasks"] for r in results)
    solved_cases = sum(r["num_subtasks"] > 0 and r["num_success_subtasks"] == r["num_subtasks"] for r in results)
    total_success_subtasks = sum(r["num_success_subtasks"] for r in results)

    sources = sorted(set(str(r.get('source', 'unknown')) for r in results))
    methods = sorted(set(str(r.get('method', 'unknown')) for r in results))
    print(f"Source(s): \033[96m{', '.join(sources)}\033[0m")
    print(f"Method(s): \033[96m{', '.join(methods)}\033[0m")
    print(f"Cases: \033[96m{total_cases}\033[0m")
    print(f"Subtasks: \033[96m{total_subtasks}\033[0m")
    print(f"Subtask success rate: \033[96m{100.0 * total_success_subtasks / max(total_subtasks, 1):.1f}%\033[0m")
    print(f"Fully solved case rate: \033[96m{100.0 * solved_cases / max(total_cases, 1):.1f}%\033[0m")

    if all_subtask_times:
        print(
            "Mean subtask wall time (all): "
            f"\033[96m{format_hms(float(np.mean(all_subtask_times)))}"
            f"±{format_hms(float(np.std(all_subtask_times)))}\033[0m"
        )
    if successful_subtask_times:
        print(
            "Mean subtask wall time (success only): "
            f"\033[96m{format_hms(float(np.mean(successful_subtask_times)))}"
            f"±{format_hms(float(np.std(successful_subtask_times)))}\033[0m"
        )
    if case_total_success_times:
        print(
            "Mean total case wall time (success only): "
            f"\033[96m{format_hms(float(np.mean(case_total_success_times)))}"
            f"±{format_hms(float(np.std(case_total_success_times)))}\033[0m"
        )
    if case_total_times:
        print(
            "Mean total case wall time (all attempts): "
            f"\033[96m{format_hms(float(np.mean(case_total_times)))}"
            f"±{format_hms(float(np.std(case_total_times)))}\033[0m"
        )
    if case_mean_success_times:
        print(
            "Mean per-case subtask wall time (success only): "
            f"\033[96m{format_hms(float(np.mean(case_mean_success_times)))}"
            f"±{format_hms(float(np.std(case_mean_success_times)))}\033[0m"
        )
    if case_mean_times:
        print(
            "Mean per-case subtask wall time (all attempts): "
            f"\033[96m{format_hms(float(np.mean(case_mean_times)))}"
            f"±{format_hms(float(np.std(case_mean_times)))}\033[0m"
        )

    failure_reasons: dict[str, int] = {}
    for r in results:
        for reason in r.get("subtask_failure_reasons", []):
            if reason is not None:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    if failure_reasons:
        print("Failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {reason}: {count}")


def default_save_dir(base_dir: str, source: InputSource, method: str) -> str:
    if source == 'task_plan_results':
        # Keep backward compatibility with previous script outputs.
        return os.path.join(base_dir, 'motion_plan_time_results', method.upper())
    return os.path.join(base_dir, 'motion_plan_time_results', 'random_grasp_pairs')


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch motion-planning timing benchmark")
    parser.add_argument('--m', type=int, default=None, help='Number of modules to filter. Default: all available.')
    parser.add_argument(
        '--source',
        type=str,
        choices=['task_plan_results', 'random_grasp_pairs'],
        default='task_plan_results',
        help='Input source directory under --dir.',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Optional explicit input directory. Overrides --dir/--source.',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Optional explicit output directory for timing results.',
    )
    parser.add_argument('--method', type=str, default='MCTS', help='Task-plan method to evaluate: IAB_BFS or MCTS. Ignored for random_grasp_pairs.')
    parser.add_argument('--f', type=str, default=None, help='Optional single filename stem, e.g. 5_100123')
    parser.add_argument('--total_budget', type=float, default=15.0, help='Atlas-RRT total time budget per motion subtask (s)')
    parser.add_argument('--optim_budget', type=float, default=20.0, help='Atlas-RRT* optimization time budget after first solution (s); ignored when --feasibility_only is set')
    parser.add_argument('--feasibility_only', action=argparse.BooleanOptionalAction, default=True, help='Stop at the first feasible Atlas-RRT solution and exclude Atlas-RRT* refinement time')
    parser.add_argument('--max_cases', type=int, default=None, help='Optional cap on number of input files to process')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False, help='Overwrite existing per-case timing results')
    parser.add_argument('--generate_mode', action=argparse.BooleanOptionalAction, default=True, help='Run motion planning and save timing results')
    parser.add_argument('--analyze_mode', action=argparse.BooleanOptionalAction, default=False, help='Only analyze saved timing results')
    default_dirname = os.path.dirname(os.path.abspath(__file__)) + '/data'
    parser.add_argument('--dir', type=str, default=default_dirname, help='Base data directory')
    args = parser.parse_args()

    source: InputSource = args.source  # type: ignore[assignment]
    method = args.method.upper()

    if source == 'task_plan_results' and method not in {'IAB_BFS', 'MCTS'}:
        raise ValueError("For --source task_plan_results, --method must be IAB_BFS or MCTS.")

    input_dir = args.input_dir or os.path.join(args.dir, source)
    save_dir = args.save_dir or default_save_dir(args.dir, source, method)
    os.makedirs(save_dir, exist_ok=True)

    if args.analyze_mode and not args.generate_mode:
        results = []
        pattern = re.compile(r'^(\d+_\d+)\.pt$')
        for p in sorted(Path(save_dir).iterdir()):
            if not pattern.match(p.name):
                continue
            if args.f is not None and p.stem != args.f:
                continue
            if args.m is not None and int(p.stem.split('_')[0]) != args.m:
                continue
            result = torch.load(p, weights_only=False)
            if result.get('source', source) != source:
                continue
            results.append(result)
        summarize_results(results)
        return

    results: list[dict] = []
    num_processed = 0

    for stem, m, seed, data in iter_saved_cases(input_dir, args.m):
        if args.f is not None and stem != args.f:
            continue
        if args.max_cases is not None and num_processed >= args.max_cases:
            break

        save_path = os.path.join(save_dir, stem + '.pt')
        if os.path.exists(save_path) and not args.overwrite:
            result = torch.load(save_path, weights_only=False)
            results.append(result)
            num_processed += 1
            print(f"{stem}: loaded cached timing result")
            continue

        try:
            gmrc_pairs = build_motion_subtasks(
                data,
                source=source,
                method=method,
                m=m,
                seed=seed,
            )
        except Exception as e:
            result = {
                'filename': stem,
                'm': m,
                'seed': seed,
                'source': source,
                'method': method if source == 'task_plan_results' else 'RANDOM_GRASP',
                'num_subtasks': 0,
                'num_success_subtasks': 0,
                'subtask_wall_times': [],
                'subtask_success': [],
                'subtask_failure_reasons': [f"Subtask extraction failed: {type(e).__name__}: {e}"],
                'case_total_wall_time': 0.0,
                'case_total_wall_time_success_only': 0.0,
                'case_mean_subtask_wall_time': None,
                'case_mean_subtask_wall_time_success_only': None,
                'feasibility_only': bool(args.feasibility_only),
            }
            torch.save(result, save_path)
            results.append(result)
            num_processed += 1
            print(f"{stem}: subtask extraction failed: {type(e).__name__}: {e}")
            continue

        if len(gmrc_pairs) == 0:
            result = {
                'filename': stem,
                'm': m,
                'seed': seed,
                'source': source,
                'method': method if source == 'task_plan_results' else 'RANDOM_GRASP',
                'num_subtasks': 0,
                'num_success_subtasks': 0,
                'subtask_wall_times': [],
                'subtask_success': [],
                'subtask_failure_reasons': [],
                'case_total_wall_time': 0.0,
                'case_total_wall_time_success_only': 0.0,
                'case_mean_subtask_wall_time': None,
                'case_mean_subtask_wall_time_success_only': None,
                'feasibility_only': bool(args.feasibility_only),
            }
            torch.save(result, save_path)
            results.append(result)
            num_processed += 1
            print(f"{stem}: no motion subtask extracted")
            continue

        print(f"{stem}: start motion planning for {len(gmrc_pairs)} subtasks [{source}]")
        subtask_times: list[float] = []
        subtask_success: list[bool] = []
        subtask_failure_reasons: list[str | None] = []

        for gmrc_src, gmrc_tgt in tqdm.tqdm(gmrc_pairs, leave=False):
            _, dt, ok, failure_reason = plan_motion_subtask(
                gmrc_src,
                gmrc_tgt,
                total_budget=args.total_budget,
                optim_budget=args.optim_budget,
                feasibility_only=args.feasibility_only,
            )
            subtask_times.append(dt)
            subtask_success.append(ok)
            subtask_failure_reasons.append(None if ok else failure_reason)

        success_times = [t for t, ok in zip(subtask_times, subtask_success) if ok]
        result = {
            'filename': stem,
            'm': m,
            'seed': seed,
            'source': source,
            'method': method if source == 'task_plan_results' else 'RANDOM_GRASP',
            'num_subtasks': len(gmrc_pairs),
            'num_success_subtasks': int(sum(subtask_success)),
            'subtask_wall_times': subtask_times,
            'subtask_success': subtask_success,
            'subtask_failure_reasons': subtask_failure_reasons,
            'case_total_wall_time': float(sum(subtask_times)),
            'case_total_wall_time_success_only': float(sum(success_times)),
            'case_mean_subtask_wall_time': float(np.mean(subtask_times)) if subtask_times else None,
            'case_mean_subtask_wall_time_success_only': float(np.mean(success_times)) if success_times else None,
            'feasibility_only': bool(args.feasibility_only),
        }
        if source == 'random_grasp_pairs' and isinstance(data, dict):
            result['action'] = data.get('action')
            result['pair_metadata'] = data.get('metadata')

        torch.save(result, save_path)
        results.append(result)
        num_processed += 1

        success_rate = 100.0 * result['num_success_subtasks'] / max(result['num_subtasks'], 1)
        mean_dt_success = result['case_mean_subtask_wall_time_success_only']
        mean_dt_all = result['case_mean_subtask_wall_time']
        mean_dt_success_str = format_hms(mean_dt_success) if mean_dt_success is not None else 'N/A'
        mean_dt_all_str = format_hms(mean_dt_all) if mean_dt_all is not None else 'N/A'
        print(
            f"{stem}: success {result['num_success_subtasks']}/{result['num_subtasks']} "
            f"({success_rate:.1f}%), mean success subtask time {mean_dt_success_str}; "
            f"mean all-attempt subtask time {mean_dt_all_str}"
        )

    if num_processed == 0:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"\033[91mInput directory does not exist: {input_dir}\033[0m")
        else:
            all_pt = sorted(input_path.glob("*.pt"))
            if not all_pt:
                print(f"\033[91mNo .pt files found in input directory: {input_dir}\033[0m")
            else:
                sample_names = ", ".join(p.name for p in all_pt[:10])
                print(f"\033[91mFound {len(all_pt)} .pt files in {input_dir}, but none matched the current filters.\033[0m")
                print(f"Sample files: {sample_names}")
                if args.m is not None:
                    print(f"Current module filter: --m {args.m}")
                if args.f is not None:
                    print(f"Current filename filter: --f {args.f}")

    summarize_results(results)


if __name__ == "__main__":
    main()
