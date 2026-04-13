import argparse
from stable_baselines3 import PPO
import pandas as pd

from Environment.GNC_Env import *


def use_model(args, outlier_ratio, pt_dir=None):
    model = PPO.load(args.model_save_path)

    test_env = GNC_Env(outlier_ratio=outlier_ratio, pt_dir=pt_dir)
    assert isinstance(test_env.action_space, gym.spaces.Box), "only continuous action space is supported"

    print("Warming up the model...")
    obs, info = test_env.reset()
    for _ in range(5):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = test_env.step(action)
    print("Warm-up complete.")


    obs, info = test_env.reset()
    
    epoch = 50
    epoch_i = 0

    data = []
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)

        if terminated or truncated:
            epoch_i += 1
            rl_gt_distance = test_env.costfactor.gt_distance()

            data.append({'Method': 'RL GNC',
                         # 'rotation_error': rl_gt_distance[0],
                         # 'translation_error': rl_gt_distance[1],
                         'rotation_error': np.log10(rl_gt_distance[0]),
                         'translation_error': np.log10(rl_gt_distance[1]),
                         'iteration': test_env.costfactor.iteration,
                         })

            if epoch_i == epoch:
                break

            obs, info = test_env.reset()

    df = pd.DataFrame(data)
    metrics = ['rotation_error', 'translation_error', 'iteration']
    for metric in metrics:
        print(f"\n=== {metric.upper()} ===")
        grouped = df.groupby('Method')[metric]
        summary = pd.DataFrame({
            'Average': grouped.mean(),
            'Median': grouped.median(),
            'Variance': grouped.var()
        })
        print(summary.to_string(formatters={
            'Average': '{:.4f}'.format,
            'Median': '{:.4f}'.format,
            'Variance': '{:.6f}'.format
        }))


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Solver_GNC_Env')
    parser.add_argument('--model-save-path', type=str, default='runs/ppo_gnc_actor.zip')
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    pt_dir = "/home/mai/python_ws/RLGNC/dataset/PointCloud_dataset/EPFL_RG-PCD/bunny.ply"
    use_model(args, outlier_ratio=0.95, pt_dir=pt_dir)