"""Tests for evaluate.py episode splitting (C2) — cheap, no simulation.

The env auto-resets inside `step()` (CONTRACT §4.5), so per-episode metrics hinge on cutting
the rollout at `info["ep_step"]` boundaries rather than at `done` + obs.  These tests pin the
split on synthetic counters, including the "episode restarts at 1, not 0" detail.
"""

import numpy as np

from evaluate import episode_arrays, split_episodes


def test_split_full_episodes():
    ep = np.array([1, 2, 3, 4, 5, 6] * 2)
    blocks = split_episodes(ep)
    assert [list(b) for b in blocks] == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]


def test_split_geofence_then_long_episode():
    # env 0: a 3-step episode (geofence) then a full 6-step one
    ep = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6])
    assert [list(b) for b in split_episodes(ep)] == [[0, 1, 2], [3, 4, 5, 6, 7, 8]]


def test_episode_arrays_all_episodes_when_unmeasured():
    rec = {"ep_step": np.array([[1], [2], [1], [2]], dtype=np.int64),
           "timeout": np.array([[0.0], [1.0], [0.0], [1.0]]),
           "geofence": np.zeros((4, 1)), "xy": np.ones((4, 1, 2))}
    ep = episode_arrays(rec, 2)                  # measured=None -> both episodes
    assert ep["complete"].tolist() == [[True], [True]]
    assert ep["xy"].shape == (2, 1, 2, 2)


def test_split_rejects_repeated_counter():
    # a stalled counter (no increment) is a boundary, never a negative-size block
    ep = np.array([1, 2, 2, 3, 4])
    blocks = split_episodes(ep)
    assert all(b.size > 0 for b in blocks)
    assert sum(b.size for b in blocks) == ep.size


def test_episode_arrays_flags_incomplete_and_pads():
    T, N, max_steps = 6, 2, 3
    # both envs: a full 3-step episode at rows 0-2, then rows 3-5 (env 1 times out there)
    rec = {
        "ep_step": np.array([[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]], dtype=np.int64),
        "timeout": np.zeros((T, N)),
        "geofence": np.zeros((T, N)),
        "xy": np.arange(T * N * 2, dtype=np.float64).reshape(T, N, 2),
    }
    rec["timeout"][2, 0] = 1.0        # env 0 finishes a full 3-step episode at t=2
    rec["timeout"][5, 1] = 1.0        # env 1's second episode ends at t=5
    ep = episode_arrays(rec, max_steps, measured=None)
    # right-aligned: episode 0 = rows 0-2 (env 0 timed out, env 1 did not),
    # episode 1 = rows 3-5 (env 0 did not time out there, env 1 did)
    assert ep["complete"].tolist() == [[True, False], [False, True]]
    assert ep["xy"].shape == (2, N, max_steps, 2)
    assert np.allclose(ep["xy"][0, 0, 2], rec["xy"][2, 0])
    assert np.allclose(ep["xy"][1, 1, 2], rec["xy"][5, 1])
    # `measured=1` keeps only the last block per env: env 0's second episode has no timeout
    assert episode_arrays(rec, max_steps, measured=1)["complete"].tolist() == [[False, True]]
    # a geofence end is not a completed episode. With measured=1 both envs contribute their
    # LAST block: env 0 has no timeout, env 1 now ends on a geofence exit -> both incomplete
    rec["timeout"][5, 1] = 0.0
    rec["geofence"][5, 1] = 1.0
    assert episode_arrays(rec, max_steps, 1)["complete"].tolist() == [[False, False]]
    # the same geofence block in the full view: only episode 0 env 0 is complete
    assert episode_arrays(rec, max_steps, None)["complete"].tolist() == [[True, False],
                                                                        [False, False]]


def test_episode_arrays_short_block_is_nan_padded_and_incomplete():
    T, N, max_steps = 4, 1, 5
    rec = {"ep_step": np.array([[1], [2], [1], [2]], dtype=np.int64),
           "timeout": np.array([[0.0], [0.0], [0.0], [0.0]]),
           "geofence": np.array([[0.0], [0.0], [1.0], [0.0]]),
           "xy": np.ones((T, N, 2))}
    ep = episode_arrays(rec, max_steps, measured=1)
    assert ep["complete"].tolist() == [[False]]
    assert np.isnan(ep["xy"][0, 0, 2]).all()
