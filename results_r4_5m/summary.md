| scenario | controller | seeds | episodes | n_episodes | n_envs | duration_s | level | radial_error_mean | radial_error_p95 | radial_error_max | cross_track_rmse | settling_time_s | settling_p95_s | energy | energy_j_proxy | action_rate | settled_pct | waypoints_reached | lap_pct | lap_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wp_square | ppo | 4 | 1 | 32 | 8 | 200 | 1 | 5.46 | 23.3 | 34.6 | 4.75 | inf | inf | 109 | 2.1e+04 | 0.293 | 3.12 | 0.844 | 21.1 | 0 |
| wp_square | pid | 4 | 1 | 32 | 8 | 200 | 1 | 24 | 38 | 41.1 | 10.8 | inf | inf | 253 | 4.86e+04 | 1.18 | 9.38 | 0.438 | 10.9 | 0 |
