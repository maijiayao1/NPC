import argparse
import os

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from Environment.GNC_Env import *


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-envs', type=int, default=20)
    parser.add_argument('--env-id', type=str, default='Solver_GNC_Env')

    # RL training related parameters
    parser.add_argument('--net-arch-policy', type=int, default=[16, 16])
    parser.add_argument('--net-arch-value', type=int, default=[16, 16])
    parser.add_argument('--ppo-steps', type=int, default=3_000_000)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--ppo-batch-size', type=int, default=128)

    parser.add_argument('--tensorboard-log', type=str, default='runs/ppo_gnc_tensorboard/')
    parser.add_argument('--model-save-path', type=str, default='runs/ppo_gnc_actor.zip')
    return parser.parse_args()


from stable_baselines3.common.callbacks import BaseCallback
class InfoLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.keys_to_log = ["err_r", "err_t", "T_iter", "mu_iter", "mu"]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            for key in self.keys_to_log:
                if key in info:
                    self.logger.record(f"rollout_ours/{key}", info[key], exclude=[])
        return True


def train_model(args):
    env = make_vec_env(
        GNC_Env,
        env_kwargs={
            'pt_dir': "/home/mai/python_ws/RLGNC/dataset/PointCloud_dataset/EPFL/aquarius_source.pcd",
            'outlier_ratio': 0.95,
            'data_num': 100,
        },
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    policy_kwargs = dict(net_arch=dict(pi=args.net_arch_policy, vf=args.net_arch_value))
    os.makedirs(args.tensorboard_log, exist_ok=True)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=True,
                batch_size=args.ppo_batch_size,
                tensorboard_log=args.tensorboard_log,
                seed=args.seed,
                device = "cuda")

    model.learn(total_timesteps=int(args.ppo_steps),
                log_interval=args.log_interval,
                tb_log_name="PPO_GNC",
                callback=InfoLoggingCallback()
                )

    model.save(args.model_save_path)
    print(f"PPO model saved to {args.model_save_path}")

    print(model.policy.log_std)


if __name__ == '__main__':
    args = load_args()
    train_model(args)