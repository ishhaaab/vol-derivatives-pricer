Standalone timing scripts mirroring the upstream `bench/bench_iv.py` style (plain
`time.perf_counter`, `sys.path` manipulation, stdout table — no pytest-benchmark dep).
Run `python bench/bench_variance_swap.py` to time the fair variance strike calculation
across grid sizes; `python bench/bench_range_accrual.py` to time the Monte Carlo range
accrual pricer across path/step configurations and read off the pv_std (MC standard
error) for each.
