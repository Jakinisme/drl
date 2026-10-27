| scenario | controller | seeds | episodes | n_episodes | n_envs | duration_s | level | radial_error_mean | radial_error_p95 | radial_error_max | cross_track_rmse | settling_time_s | settling_p95_s | energy | energy_j_proxy | action_rate | settled_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk_wind | ppo | 4 | 1 | 31 | 8 | 120 | 1 | 12.8 | 34.9 | 38.9 | 15.2 | inf | inf | 134 | 2.58e+04 | 0.552 | 28.6 |
| sk_wind | pid | 4 | 1 | 29 | 8 | 120 | 1 | 37.1 | 51 | 52.7 | 37.8 | inf | inf | 223 | 4.28e+04 | 1.07 | 0 |
| wp_square | ppo | 4 | 1 | 32 | 8 | 200 | 1 | 4.46 | 21.9 | 34.3 | 3.97 | inf | inf | 88.7 | 1.7e+04 | 0.906 | 25 |
| wp_square | pid | 4 | 1 | 31 | 8 | 200 | 1 | 26 | 40.6 | 45 | 12.4 | inf | inf | 252 | 4.84e+04 | 1.2 | 16.5 |
| rough_sea | ppo | 4 | 1 | 25 | 8 | 120 | 1 | 12.8 | 34.3 | 38.6 | 17.1 | inf | inf | 166 | 3.2e+04 | 0.514 | 20.7 |
| rough_sea | pid | 4 | 1 | 20 | 8 | 120 | 1 | 37.3 | 51.1 | 52.7 | 38.7 | inf | inf | 236 | 4.54e+04 | 0.824 | 0 |
