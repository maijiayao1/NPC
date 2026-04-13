import numpy as np
import open3d as o3d


def skew(x):
    x = x.reshape(-1)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def skew_batch(pts):
    S_skew = np.zeros((pts.shape[0], 3, 3))
    S_skew[:, 0, 1] = -pts[:, 2]
    S_skew[:, 1, 0] = pts[:, 2]
    S_skew[:, 0, 2] = pts[:, 1]
    S_skew[:, 2, 0] = -pts[:, 1]
    S_skew[:, 1, 2] = -pts[:, 0]
    S_skew[:, 2, 1] = pts[:, 0]

    return S_skew


def se3_exp(xi):
    xi = xi.reshape(-1)
    omega = xi[:3]
    v = xi[3:]

    theta = np.linalg.norm(omega)
    T = np.eye(4)

    if theta < 1e-10:
        R = np.eye(3)
        t = v
    else:
        n = (omega / theta).reshape(-1, 1)
        R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * n @ n.T + np.sin(theta) * skew(n)
        J = np.sin(theta) * np.eye(3) + (theta - np.sin(theta)) * n @ n.T + (1 - np.cos(theta)) * skew(n)
        t = J @ v

    T[:3, :3] = R
    T[:3, 3] = t

    return T


def normalize_unit_cube(point_cloud):
    points = point_cloud[:, :3]
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    scale = max(max_xyz - min_xyz)
    normalized_xyz = (points - min_xyz) / scale

    normalized_point_cloud = point_cloud.copy()
    normalized_point_cloud[:, :3] = normalized_xyz
    return normalized_point_cloud


def random_sampling(points, num_samples, rng=None):
    if num_samples >= points.shape[0]:
        print(f"warning: num_samples >= points.size!! ({num_samples} vs {points.shape[0]})")
        return points.copy()

    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(points.shape[0], size=num_samples, replace=False)
    return points[indices]


def read_fuse(filename):
    file = open(filename).readlines()
    pt = []
    for line in file[:]:
        nextline = []
        lines = line.split(" ")
        nextline.append(float(lines[0]))
        nextline.append(float(lines[1]))
        nextline.append(float(lines[2]))
        nextline.append(float(1.0))
        pt.append(nextline)
    pt = np.array(pt)
    return pt

def read_pt(filename):
    pcd = o3d.io.read_point_cloud(filename)
    pt = np.asarray(pcd.points)
    if pt.shape[1] == 3:
        homogeneous = np.ones((pt.shape[0], 1))
        pt = np.hstack((pt, homogeneous))
    return pt