"""
Generate random single-grasp configuration pairs for motion-planning benchmarks.

This script is meant to be much cheaper than running full task planning.
For each successful case, it:
  1) samples a random GMRC configuration,
  2) enumerates all currently feasible actions,
  3) keeps only grasp actions,
  4) randomly chooses one grasp action that can be executed successfully,
  5) saves the (before, after) GMRC pair to disk.

The saved files can then be consumed by a dedicated motion-planning timing script,
or inspected manually. The output format is a torch-saved dictionary containing:
  - gmrc_1: configuration before the random grasp action
  - gmrc_2: configuration after the random grasp action
  - action: the selected grasp action object
  - metadata: seed / counts / trial information
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append('..')

from GMRC import GMRC


def get_random_grasp_actions(gmrc: GMRC) -> list[Any]:
    """Return all candidate grasp actions for the current configuration."""
    return [action for action in gmrc.get_all_actions() if isinstance(action, tuple)]


def build_random_grasp_pair(
    m: int,
    seed: int,
    *,
    max_action_trials: int | None = None,
    action_seed_offset: int = 10_000_000,
) -> dict[str, Any] | None:
    """
    Try to build one valid (gmrc_1, gmrc_2) pair connected by a single random
    grasp action. Returns None if the sampled base configuration fails or if no
    grasp action can be executed successfully.
    """
    gmrc_1 = GMRC.get_random_configuration(m=m, seed=seed)
    if not gmrc_1.successfully_spawned:
        return None

    grasp_actions = get_random_grasp_actions(gmrc_1)
    if len(grasp_actions) == 0:
        return None

    rng = np.random.default_rng(action_seed_offset + seed)
    perm = rng.permutation(len(grasp_actions)).tolist()
    if max_action_trials is not None:
        perm = perm[: max(1, max_action_trials)]

    for trial_id, action_idx in enumerate(perm):
        gmrc_2 = gmrc_1.copy()
        action = grasp_actions[action_idx]
        success = bool(gmrc_2.execute_action(action))
        if not success:
            continue

        # execute_action() already updates angles / geometry / colliders, but we
        # keep the collision guard explicit to avoid silently saving bad pairs.
        if gmrc_2.is_collision_detected():
            continue

        return {
            "gmrc_1": gmrc_1,
            "gmrc_2": gmrc_2,
            "action": action,
            "metadata": {
                "m": m,
                "seed": seed,
                "num_candidate_grasp_actions": len(grasp_actions),
                "selected_action_index": int(action_idx),
                "action_trial_rank": int(trial_id),
            },
        }

    return None


def iter_saved_pairs(save_dir: Path, m: int | None = None):
    for path in sorted(save_dir.glob("*.pt")):
        stem = path.stem
        try:
            cur_m_str, seed_str = stem.split("_", 1)
            cur_m = int(cur_m_str)
            _ = int(seed_str)
        except ValueError:
            continue
        if m is not None and cur_m != m:
            continue
        yield path, torch.load(path, weights_only=False)


def print_summary(records: list[dict[str, Any]]) -> None:
    if not records:
        print("\033[91mNo random grasp-pair records found.\033[0m")
        return

    num_actions = [
        rec["metadata"]["num_candidate_grasp_actions"]
        for rec in records
        if "metadata" in rec and "num_candidate_grasp_actions" in rec["metadata"]
    ]
    trial_ranks = [
        rec["metadata"]["action_trial_rank"]
        for rec in records
        if "metadata" in rec and "action_trial_rank" in rec["metadata"]
    ]

    print(f"Saved cases: \033[96m{len(records)}\033[0m")
    if num_actions:
        print(
            "Mean candidate grasp actions: "
            f"\033[96m{float(np.mean(num_actions)):.2f}"
            f" ± {float(np.std(num_actions)):.2f}\033[0m"
        )
    if trial_ranks:
        print(
            "Mean selected-action trial rank: "
            f"\033[96m{float(np.mean(trial_ranks)):.2f}"
            f" ± {float(np.std(trial_ranks)):.2f}\033[0m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random single-grasp GMRC pairs for motion-planning tests"
    )
    parser.add_argument('--m', type=int, required=True, help='Number of modules')
    parser.add_argument('--n', type=int, default=100, help='Number of successful pairs to save')
    parser.add_argument('--s', type=int, default=100000, help='Starting seed')
    parser.add_argument(
        '--max_action_trials',
        type=int,
        default=None,
        help='Optional cap on how many randomly ordered grasp actions to try per base configuration',
    )
    parser.add_argument(
        '--max_seed_tries',
        type=int,
        default=100000,
        help='Safety cap on total seeds examined while trying to collect n successful pairs',
    )
    parser.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Overwrite an existing saved pair file if present',
    )
    parser.add_argument(
        '--generate_mode',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate and save new pair data',
    )
    parser.add_argument(
        '--analyze_mode',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Load saved pair data and print summary statistics',
    )
    default_dirname = os.path.dirname(os.path.abspath(__file__)) + '/data'
    parser.add_argument('--dir', type=str, default=default_dirname, help='Base data directory')
    args = parser.parse_args()

    save_dir = Path(args.dir) / 'random_grasp_pairs'
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_mode:
        GMRC.suppress_spawn_err = True
        GMRC.suppress_action_err = True

        success_count = 0
        checked_count = 0
        seed = args.s

        print(f"\033[94mm = {args.m}; Start generating {args.n} random grasp-pair cases\033[0m")

        while success_count < args.n and checked_count < args.max_seed_tries:
            out_path = save_dir / f"{args.m}_{seed}.pt"
            checked_count += 1

            if out_path.exists() and not args.overwrite:
                print(f"Seed {seed}: \033[33mexists, skipped\033[0m")
                success_count += 1
                seed += 1
                continue

            pair_data = build_random_grasp_pair(
                args.m,
                seed,
                max_action_trials=args.max_action_trials,
            )

            if pair_data is None:
                print(f"Seed {seed}: \033[91mfailed\033[0m")
                seed += 1
                continue

            torch.save(pair_data, out_path)
            md = pair_data["metadata"]
            print(
                f"Seed {seed}: \033[92msaved\033[0m; "
                f"candidate grasp actions = {md['num_candidate_grasp_actions']}; "
                f"selected trial rank = {md['action_trial_rank']}"
            )
            success_count += 1
            seed += 1

        print(
            f"\nFinished: saved \033[96m{success_count}\033[0m cases "
            f"after checking \033[96m{checked_count}\033[0m seeds."
        )

    if args.analyze_mode:
        records = [record for _, record in iter_saved_pairs(save_dir, args.m)]
        print_summary(records)


if __name__ == '__main__':
    main()
