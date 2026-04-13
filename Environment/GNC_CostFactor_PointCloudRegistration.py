import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import entropy

from Environment.point_cloud_registration_utils import *

seed = 48
rng = np.random.default_rng(seed)

class CostFactor_pointCloud:
    def __init__(self, T_gt, T_init=np.eye(4), pt_raw=None, data_num=100, outlier_ratio=0.95,
                 robust_kernel=None, gnc_factor=1.4):
        self.T_gt = T_gt
        self.T = T_init
        self.T_prev = self.T
        self.data_num = data_num
        self.data_dim = 4
        self.outlier_ratio = outlier_ratio
        self.source_pt, self.target_pt = self.make_noisy_test_data(pt_raw)

        # Solver related params
        self.robust_kernel = robust_kernel
        self.gnc_factor = gnc_factor

        self.iteration = 0
        self.mu_iter = 0
        self.max_outer_iter = 300
        self.max_inner_iter = 100
        self.all_iter = self.max_outer_iter * self.max_inner_iter

        self.tolDeltaX = 1e-3
        self.noise_bound = 1e-3

        self.mu_iter_hst = np.zeros((self.max_outer_iter + 1, 3))

        self.DoF = self.data_num - 5 # self.DoF = self.data_num - 6 + 1

        self.jacobian_ = self.jacobian()
        self.residual_ = self.residual()
        self.max_abs_residual = self.max_residual()

        self.mu = min(2 * np.max(np.square(self.residual_)) / self.noise_bound, 1e6)
        self.weights_ = self.GNC_GM_weight_update()

        self.mu_iter_hst[self.mu_iter, 0] = self.target_f()

        self.done = False


    def make_noisy_test_data(self, pt_raw):
        if pt_raw is None:
            pt_dir = "/home/mai/python_ws/RLGNC/dataset/PointCloud_dataset/EPFL/aquarius_source.pcd"
            pt_raw = read_pt(pt_dir)

        source_pt = random_sampling(points=pt_raw, num_samples=self.data_num, rng=rng)
        source_pt = normalize_unit_cube(source_pt)

        target_pt = source_pt @ self.T_gt.T
        target_pt[:, 0:3] += rng.normal(loc=0.0, scale=0.01, size=(self.data_num, 3))

        self.outlier_id = np.ones(self.data_num)
        if self.outlier_ratio > 0:
            outlier_num = np.ceil(self.data_num * self.outlier_ratio)
            for _ in range(int(outlier_num)):
                rand_row = rng.integers(0, self.data_num, size=1)
                source_pt[rand_row, 0:3] += rng.uniform(-1, 1, size=(1,3))
                self.outlier_id[rand_row] = 0
        return source_pt, target_pt


    def residual(self):
        r = self.target_pt - self.source_pt @ self.T.T
        r = r.reshape(self.data_num * self.data_dim, 1)
        return r


    def max_residual(self):
        r_abs = np.abs(self.residual_)
        residuals = r_abs.reshape(self.data_num, 4)
        residuals = np.amax(residuals, axis=1)
        residuals = np.repeat(residuals, 4)
        residuals = residuals.reshape(self.data_num * self.data_dim, 1)
        return residuals


    def jacobian(self):
        J = np.zeros((self.data_num, self.data_dim, 6))
        S_skew = skew_batch(self.source_pt[:, :3])
        R_x_S_skew = self.T[:3, :3] @ S_skew
        J[:, :3, :3] = R_x_S_skew
        J[:, :3, 3:6] = -self.T[:3, :3]
        return J.reshape(self.data_num * self.data_dim, 6)

    def hessian(self):
        if self.weights_.shape != (self.data_num * self.data_dim, 1):
            self.weights_ = np.repeat(self.weights_, self.data_dim, axis=1)
            self.weights_ = self.weights_.reshape(self.data_num * self.data_dim, 1)
        return self.jacobian_.T @ ( self.jacobian_ * self.weights_ )


    def gradient(self):
        if self.weights_.shape != (self.data_num * self.data_dim, 1):
            self.weights_ = np.repeat(self.weights_, self.data_dim, axis=1)
            self.weights_ = self.weights_.reshape(self.data_num * self.data_dim, 1)
        return self.jacobian_.T @ ( self.weights_ * self.residual_ )


    def target_f(self):
        r_sq = self.residual_**2
        rho = self.noise_bound * r_sq / (r_sq + self.noise_bound)
        return np.linalg.norm(rho)


    def robust_kernel_weights_factory(self, param_tuple=None):
        residual_sq = np.square(self.max_abs_residual)

        if self.robust_kernel == "Cauchy":
            if param_tuple is None:
                param_tuple = {"c": 0.1}
            return 1 / (1 + residual_sq / np.square(param_tuple["c"]))
        elif self.robust_kernel == "Geman-McClure":
            c_sq = 4
            return residual_sq / (c_sq + residual_sq)
        else:
            return np.ones((self.data_num * self.data_dim, 1))


    def GNC_GM_weight_update(self):
        residual_sq = np.square(self.max_abs_residual)
        residual_sq = np.clip(residual_sq, -1e8, 1e8)
        weights = np.square(self.mu * self.noise_bound / (residual_sq + self.mu * self.noise_bound))
        weights = np.clip(weights, 0, 1)
        return weights


    def update_step(self, action):
        self.mu_iter += 1
        self.mu /= action[0]
        self.weights_ = self.GNC_GM_weight_update()
        delta_T, inner_iter = self.solve(10 ** action[1])

        self.mu_iter_hst[self.mu_iter, 0] = self.target_f()
        self.mu_iter_hst[self.mu_iter, 1] = np.linalg.norm(delta_T)
        self.mu_iter_hst[self.mu_iter, 2] = inner_iter

        if self.mu < 1:
            self.done = True
            return self.done

        if self.mu_iter >= self.max_outer_iter:
            self.done = True
            return self.done

        self.done = False
        return self.done


    def solve(self, tolDeltaX=None):
        inner_iter = 0
        while True:
            self.iteration += 1
            inner_iter += 1

            H = self.hessian() + 1e-12 * np.eye(6)
            g = self.gradient()
            delta_T = np.linalg.solve(H, -g)

            # update
            self.T = self.T @ se3_exp(delta_T)

            self.jacobian_ = self.jacobian()
            self.residual_ = self.residual()
            self.max_abs_residual = self.max_residual()

            if tolDeltaX is None:
                tolDeltaX = self.tolDeltaX

            if np.linalg.norm(delta_T) < tolDeltaX:
                return delta_T, inner_iter

            if inner_iter == self.max_inner_iter:
                return delta_T, inner_iter


    def gt_distance(self):
        err_T = np.linalg.inv(self.T_gt) @ self.T
        err_rot = np.arccos((np.trace(err_T[0:3, 0:3]) - 1) / 2) * 180 / np.pi
        err_t = np.linalg.norm(err_T[0:3, 3])
        return err_rot, err_t