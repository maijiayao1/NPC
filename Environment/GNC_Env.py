import gymnasium as gym
from Environment.GNC_CostFactor_PointCloudRegistration import *

class GNC_Env(gym.Env):
    def __init__(self, data_num=1000, outlier_ratio=0.95, pt_raw=None, pt_dir=None):
        super(GNC_Env, self).__init__()
        if pt_raw is None:
            pt_raw = read_pt(pt_dir)
        self.pt_raw = pt_raw

        self.data_num = data_num
        self.outlier_ratio = outlier_ratio
        self.lambda_1 = 1000
        self.lambda_2 = 0.001

        self.action_space = gym.spaces.Box(low=np.array([1.0, -5.0]), high=np.array([100.0, 0.0]), dtype=np.float64)
        obs_low = np.array(
                        [0.0] +     # mu
                        [-5.0] +    # log10(tolDeltaX)
                        [0] +       # iter
                        [0.0],      # delta_y
                        dtype=np.float64)
        obs_high = np.array(
                        [1e8] +     # mu
                        [0.0] +     # log10(tolDeltaX)
                        [100] +     # iter
                        [1e7],      # delta_y
                        dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        T_se3 = np.array([
            rng.uniform(-1, 1, size=1),
            rng.uniform(-1, 1, size=1),
            rng.uniform(-1, 1, size=1),
            rng.uniform(-1, 1, size=1),
            rng.uniform(-1, 1, size=1),
            rng.uniform(-1, 1, size=1),
        ])
        T_gt = se3_exp(T_se3)
        self.costfactor = CostFactor_pointCloud(T_gt=T_gt, pt_raw=self.pt_raw, data_num=self.data_num, outlier_ratio=self.outlier_ratio)


        self.costfactor.update_step(action=np.array([1.0, -1.0]))

        curr_iter = self.costfactor.mu_iter
        curr_f = self.costfactor.mu_iter_hst[curr_iter-1 + 1:curr_iter + 1, 0].reshape(-1)
        prev_f = self.costfactor.mu_iter_hst[curr_iter-1:curr_iter, 0].reshape(-1)
        obs = np.concatenate([
            np.array([self.costfactor.mu]).reshape(-1),
            np.array([np.log10(self.costfactor.tolDeltaX)]).reshape(-1),
            self.costfactor.mu_iter_hst[curr_iter:curr_iter + 1, 2].reshape(-1),
            curr_f - prev_f,
        ])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs, {}


    def step(self, action):
        self.costfactor.update_step(action)

        curr_iter = self.costfactor.mu_iter
        curr_f = self.costfactor.mu_iter_hst[curr_iter:curr_iter + 1, 0].reshape(-1)
        prev_f = self.costfactor.mu_iter_hst[curr_iter-1:curr_iter, 0].reshape(-1)
        obs = np.concatenate([
            np.array([self.costfactor.mu]).reshape(-1),
            np.array([action[1]]).reshape(-1),
            self.costfactor.mu_iter_hst[curr_iter:curr_iter + 1, 2].reshape(-1),
            curr_f - prev_f,
        ])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        flag = {"Terminated": False, "Truncated": False}
        info = {}

        delta_f = self.costfactor.mu_iter_hst[curr_iter, 0] - self.costfactor.mu_iter_hst[curr_iter-1, 0]
        reward = -delta_f * self.lambda_1
        reward = np.clip(reward, -10, 10)

        if self.costfactor.mu < 1:
            flag["Terminated"] = True
            info = self._get_info()
            reward += (self.costfactor.all_iter - self.costfactor.iteration) * self.lambda_2

        if self.costfactor.mu_iter >= self.costfactor.max_outer_iter:
            flag["Truncated"] = True
            info = self._get_info()

        return obs, float(reward), flag["Terminated"], flag["Truncated"], info


    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def _get_info(self):
        err_r, err_t = self.costfactor.gt_distance()
        return {"err_r": float(err_r),
                "err_t": float(err_t),
                "T_iter": int(self.costfactor.iteration),
                "mu_iter": int(self.costfactor.mu_iter),
                "mu": float(self.costfactor.mu),
        }