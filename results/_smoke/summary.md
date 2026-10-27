| scenario | controller | seed | episodes | n_episodes | n_envs | duration_s | wall_s | radial_error_mean | radial_error_p95 | radial_error_max | cross_track_rmse | settling_time_s | settling_p95_s | energy | energy_j_proxy | action_rate | settled_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk_wind | ppo | 0 | 1 | 4 | 4 | 120 | 77.1 | 8.7 | 31.1 | 33.6 | 13.2 | inf | inf | 169 | 3.24e+04 | 0.405 | 25 |
| sk_wind | pid | 10 | 1 | 3 | 4 | 120 | 81.3 | 34.9 | 49 | 51.1 | 36.2 | inf | inf | 213 | 4.09e+04 | 1.05 | 33.3 |
