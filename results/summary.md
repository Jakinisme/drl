# C2 — PPO vs PID, identical seeds

Checkpoint: `runs/checkpoints/ckpt_last.pt` (steps 19,988,480, iteration 305).
Env: 8 envs per seed; disturbance pinned to the top `scale_by_level` row (1.0).
Energy is in **normalised command units** (CONTRACT §4.9 left joule-equivalent open;
decided here — `energy_j_proxy` is the rough joule figure and is not the headline).

| scenario | controller | seeds | episodes | radial mean | p95 | cross-track | energy | action rate | settled |
|---|---|---|---|---|---|---|---|---|---|
| sk_wind | PPO | 4 | 31 | 12.72 m | 34.74 m | 14.99 m | 134.5 | 0.549/s | 32.3 % |
| sk_wind | PID | 4 | 29 | 37.71 m | 50.90 m | 38.13 m | 223.5 | 1.050/s | 0.0 % |
| wp_square | PPO | 4 | 32 | 4.62 m | 21.97 m | 4.03 m | 90.9 | 0.842/s | 15.6 % |
| wp_square | PID | 4 | 30 | 27.08 m | 41.71 m | 12.93 m | 251.9 | 1.202/s | 13.3 % |
| rough_sea | PPO | 1 | 6 | 6.84 m | 25.75 m | 10.66 m | 146.3 | 0.460/s | 16.7 % |
| rough_sea | PID | 1 | 5 | 32.21 m | 45.21 m | 33.49 m | 243.4 | 0.727/s | 0.0 % |

## Not a fair comparison yet

The PID baseline cannot close the loop in this sim (Wave 3 finding: every body command
yields the same surge force, so heading saturates and limit-cycles). PID settling 0.0 % on
`sk_wind` and `rough_sea` is that defect, not necessarily poor tuning. The radial gap below
therefore measures the modelling defect as much as the policy.

## Other caveats

- Placeholders: reward weights, geofence 80 m, arrival radius 2.0 m (T7/T8).
- `obs.clip_norm` ±5 at a 10 m scale hides errors beyond ~50 m.
- Episode counts under 8/8 are geofence truncations, not completions (`complete` flag).
- Nothing here was validated on the real vessel.
