uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole >> output_logs/run_small_noargs.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg >> output_logs/run_small_rtg.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na >> output_logs/run_small_na.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na >> output_logs/run_small_rtg_na.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb >> output_logs/run_large_noargs.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg >> output_logs/run_large_rtg.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na >> output_logs/run_large_na.txt
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na >> output_logs/run_large_rtg_na.txt