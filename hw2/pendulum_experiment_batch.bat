@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum >> output_logs/pendulum_output.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 --exp_name pendulum_b3000 >> output_logs/pendulum_output_b3000.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg --discount 0.95 --exp_name pendulum_b1000_rtg_disc095 >> output_logs/pendulum_output_b1000_rtg_disc095.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg --discount 0.95 --exp_name pendulum_b3000_rtg_disc095 >> output_logs/pendulum_output_b3000_rtg_disc095.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg --discount 0.95 --exp_name pendulum_b5000_rtg_disc095 >> output_logs/pendulum_output_b5000_rtg_disc095.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 7000 -eb 1000 -rtg --discount 0.95 --exp_name pendulum_b7000_rtg_disc095 >> output_logs/pendulum_output_b7000_rtg_disc095.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg --discount 0.90 --exp_name pendulum_b1000_rtg_disc090 >> output_logs/pendulum_output_b1000_rtg_disc090.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg --discount 0.90 --exp_name pendulum_b3000_rtg_disc090 >> output_logs/pendulum_output_b3000_rtg_disc090.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg --discount 0.90 --exp_name pendulum_b5000_rtg_disc090 >> output_logs/pendulum_output_b5000_rtg_disc090.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 7000 -eb 1000 -rtg --discount 0.90 --exp_name pendulum_b7000_rtg_disc090 >> output_logs/pendulum_output_b7000_rtg_disc090.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg -na --discount 0.95 --exp_name pendulum_b1000_rtg_disc095_na >> output_logs/pendulum_output_b1000_rtg_disc095_na.txt
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 500 -eb 2000 -rtg -na --discount 0.95 --exp_name pendulum_b500_rtg_disc095_na >> output_logs/pendulum_output_b500_rtg_disc095_na.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 200 -eb 1000 -rtg -na --discount 0.95 --exp_name pendulum_b200_rtg_disc095_na >> output_logs/pendulum_output_b200_rtg_disc095_na.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -na --discount 0.95 --exp_name pendulum_b3000_rtg_disc095_na >> output_logs/pendulum_output_b3000_rtg_disc095_na.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -na --discount 0.95 --exp_name pendulum_b5000_rtg_disc095_na >> output_logs/pendulum_output_b5000_rtg_disc095_na.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 7000 -eb 1000 -rtg -na --discount 0.95 --exp_name pendulum_b7000_rtg_disc095_na >> output_logs/pendulum_output_b7000_rtg_disc095_na.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg -na --discount 0.95 --gae_lambda 0.99 --exp_name pendulum_b1000_rtg_disc095_na_gae099 >> output_logs/pendulum_output_b1000_rtg_disc095_na_gae099.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -na --discount 0.95 --gae_lambda 0.99 --exp_name pendulum_b3000_rtg_disc095_na_gae099 >> output_logs/pendulum_output_b3000_rtg_disc095_na_gae099.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -na --discount 0.95 --gae_lambda 0.99 --exp_name pendulum_b5000_rtg_disc095_na_gae099 >> output_logs/pendulum_output_b5000_rtg_disc095_na_gae099.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 7000 -eb 1000 -rtg -na --discount 0.95 --gae_lambda 0.99 --exp_name pendulum_b7000_rtg_disc095_na_gae099 >> output_logs/pendulum_output_b7000_rtg_disc095_na_gae099.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg -na --discount 0.95 -lr 3e-3 --gae_lambda 1 --exp_name pendulum_b1000_rtg_disc095_na_gae099_lr3e3 >> output_logs/pendulum_output_b1000_rtg_disc095_na_gae099_lr3e3.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -na --discount 0.95 -lr 3e-3 --gae_lambda 1 --exp_name pendulum_b3000_rtg_disc095_na_gae099_lr3e3 >> output_logs/pendulum_output_b3000_rtg_disc095_na_gae099_lr3e3.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -na --discount 0.95 -lr 3e-3 --gae_lambda 1 --exp_name pendulum_b5000_rtg_disc095_na_gae099_lr3e3 >> output_logs/pendulum_output_b5000_rtg_disc095_na_gae099_lr3e3.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 7000 -eb 1000 -rtg -na --discount 0.95 -lr 3e-3 --gae_lambda 1 --exp_name pendulum_b7000_rtg_disc095_na_gae099_lr3e3 >> output_logs/pendulum_output_b7000_rtg_disc095_na_gae099_lr3e3.txt

@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -lr 1e-3 --exp_name pendulum_rtg_lr1e3_b3000 >> output_logs/pendulum_rtg_lr1e3_b3000_output.txt 
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -lr 1e-3 -na --use_baseline --exp_name pendulum_rtg_na_base_lr1e3_b3000 >> output_logs/pendulum_rtg_na_baseline_lr1e3_b3000_output.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 3000 -eb 1000 -rtg -lr 1e-3 -na --use_baseline --gae_lambda 1 --exp_name pendulum_rtg_na_base_lr1e3_b3000_gae1 >> output_logs/pendulum_rtg_na_baseline_lr1e3_b3000_gae1_output.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -lr 1e-3 --exp_name pendulum_rtg_lr1e3_b5000 >> output_logs/pendulum_rtg_lr1e3_b5000_output.txt 
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -lr 1e-3 -na --use_baseline --exp_name pendulum_rtg_na_base_lr1e3_b5000 >> output_logs/pendulum_rtg_na_baseline_lr1e3_b5000_output.txt
@REM uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 -rtg -lr 1e-3 -na --use_baseline --gae_lambda 1 --exp_name pendulum_rtg_na_base_lr1e3_b5000_gae1 >> output_logs/pendulum_rtg_na_baseline_lr1e3_b5000_gae1_output.txt
