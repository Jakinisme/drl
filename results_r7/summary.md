| scenario | controller | seeds | episodes | n_episodes | n_envs | duration_s | level | radial_error_mean | radial_error_p95 | radial_error_max | cross_track_rmse | settling_time_s | settling_p95_s | energy | energy_j_proxy | action_rate | settled_pct | waypoints_reached | lap_pct | lap_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wp_square | ppo | 4 | 1 | 32 | 8 | 200 | 1 | 13 | 30.2 | 40.3 | 1.93 | inf | inf | 342 | 6.57e+04 | 0.505 | 53.1 | 3.78 | 94.5 | 0.844 |
| wp_square | pid | 4 | 1 | 32 | 8 | 200 | 1 | 27.1 | 38.9 | 40.8 | 10.3 | inf | inf | 247 | 4.74e+04 | 1.19 | 0 | 0.406 | 10.2 | 0 |
| sk_wind | ppo | 4 | 1 | 32 | 8 | 120 | 1 | 12.7 | 38.3 | 41.1 | 12.2 | inf | inf | 200 | 3.84e+04 | 0.509 | 43.8 | inf | inf | 0 |
| sk_wind | pid | 4 | 1 | 28 | 8 | 120 | 1 | 33.5 | 45.4 | 46.9 | 32.3 | inf | inf | 199 | 3.82e+04 | 1.09 | 4.17 | inf | inf | 0 |
| rough_sea | ppo | 4 | 1 | 23 | 8 | 120 | 1 | 13.1 | 34.2 | 36.8 | 15 | inf | inf | 212 | 4.08e+04 | 0.463 | 40 | inf | inf | 0 |
| rough_sea | pid | 4 | 1 | 18 | 8 | 120 | 1 | 37.5 | 51.6 | 53.3 | 38.9 | inf | inf | 228 | 4.39e+04 | 0.935 | 0 | inf | inf | 0 |
