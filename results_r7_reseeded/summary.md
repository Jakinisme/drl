| scenario | controller | seeds | episodes | n_episodes | n_envs | duration_s | level | radial_error_mean | radial_error_p95 | radial_error_max | cross_track_rmse | settling_time_s | settling_p95_s | energy | energy_j_proxy | action_rate | settled_pct | waypoints_reached | lap_pct | lap_completed | sat_frac | reversal_rate | arrival_speed_mean_mps | hold_rms_m | hold_p95_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk_wind | ppo | 4 | 1 | 32 | 8 | 120 | 1 | 19 | 45.7 | 48.4 | 11 | inf | inf | 205 | 3.94e+04 | 0.388 | 15.6 | inf | inf | 0 | 0.84 | 0 | inf | inf | inf |
| sk_wind | pid | 4 | 1 | 31 | 8 | 120 | 1 | 35.6 | 45.4 | 47.1 | 35.1 | inf | inf | 177 | 3.39e+04 | 1.01 | 0 | inf | inf | 0 | 0.761 | 0 | inf | inf | inf |
| wp_square | ppo | 4 | 1 | 31 | 8 | 300 | 1 | 8.97 | 29.7 | 44.2 | 1.28 | inf | inf | 530 | 1.02e+05 | 0.561 | 73.7 | 3.97 | 99.2 | 0.969 | 0.753 | 0 | 1.03 | inf | inf |
| wp_square | pid | 4 | 1 | 26 | 8 | 300 | 1 | 20.2 | 32.5 | 41.8 | 8.19 | inf | inf | 366 | 7.03e+04 | 1.18 | 0 | 1.01 | 25.3 | 0 | 0.688 | 0 | 0.488 | inf | inf |
