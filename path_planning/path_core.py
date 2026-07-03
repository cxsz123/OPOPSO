import numpy as np
from scipy.spatial.distance import cdist


def fitness(x_way, threat_interp_num, uav_num, begin_point, final_point,
            x_map, y_map, z_map, x_max, y_max, z_max, hemisphere_params, cylinder_params):
    """
    适应度函数：计算多无人机路径的综合代价
    """
    total_dim = len(x_way)
    n_dim_per_uav = total_dim // uav_num

    total_cost = 0.0
    bezier_points_all = []
    way_all = []
    path_costs = np.zeros(uav_num)
    way_penalty = 0.0

    hemi_centers = hemisphere_params[:, :3] if hemisphere_params.size > 0 else np.empty((0, 3))
    hemi_r = hemisphere_params[:, 3] if hemisphere_params.size > 0 else np.empty(0)
    hemi_safe_r = hemi_r + 8.0 if hemi_r.size > 0 else np.empty(0)

    cyl_xy = cylinder_params[:, :2] if cylinder_params.size > 0 else np.empty((0, 2))
    cyl_r = cylinder_params[:, 2] if cylinder_params.size > 0 else np.empty(0)
    cyl_safe_r = cyl_r + 8.0 if cyl_r.size > 0 else np.empty(0)

    for k in range(uav_num):
        start_idx = k * n_dim_per_uav
        end_idx = start_idx + n_dim_per_uav
        x_way_uav = x_way[start_idx:end_idx]

        bezier_points, way = generate_smooth_path(
            x_way_uav, threat_interp_num, begin_point, final_point, k + 1
        )

        bezier_points_all.append(bezier_points)
        way_all.append(way)

        obstacle_penalty = 0.0
        m = bezier_points.shape[0]

        # ========== 半球形障碍物惩罚 ==========
        if hemi_centers.size > 0 and m > 0:
            dist_3d = np.sqrt(np.sum((bezier_points[:, None] - hemi_centers[None]) ** 2, axis=2))
            # 计算路径点到障碍物表面的距离
            dist_surface = dist_3d - hemi_r[None, :]
            # 软惩罚
            mask_near = (dist_surface > 1e-8) & (dist_surface < 8.0)
            if np.any(mask_near):
                temp = dist_surface[mask_near]
                obstacle_penalty += np.sum((8.0 / temp) ** 2) * 100

            # 硬惩罚
            mask_inside = dist_3d < hemi_r[None]
            inside_count = np.sum(mask_inside)
            if inside_count > 0:
                obstacle_penalty += inside_count * 1e8

        # ========== 圆柱形障碍物惩罚 ==========
        if cyl_xy.size > 0 and m > 0:
            dist_xy = np.sqrt(np.sum((bezier_points[:, :2, None] - cyl_xy.T[None]) ** 2, axis=1))
            # 计算路径点到圆柱侧面的水平距离
            dist_surface_xy = dist_xy - cyl_r[None, :]

            # 软惩罚
            mask_near = (dist_surface_xy > 1e-8) & (dist_surface_xy < 8.0)
            if np.any(mask_near):
                temp = dist_surface_xy[mask_near]
                obstacle_penalty += np.sum((8.0 / temp) ** 2) * 100

            # 硬惩罚
            mask_inside = dist_xy < cyl_r[None]
            inside_count = np.sum(mask_inside)
            if inside_count > 0:
                obstacle_penalty += inside_count * 1e8

        N_way = way.shape[0]
        if N_way >= 2:
            segment_dist_way = np.linalg.norm(np.diff(way, axis=0), axis=1)
            threshold_way = 160
            excess_dist_way = segment_dist_way[segment_dist_way > threshold_way] - threshold_way
            way_penalty += np.sum(excess_dist_way) ** 1.5

        path_cost = compute_single_path_cost(bezier_points, x_map, y_map, z_map)
        path_cost += obstacle_penalty
        path_costs[k] = path_cost

    total_path_cost = np.sum(path_costs)
    coop_cost = compute_cooperation_cost(bezier_points_all, uav_num)
    total_cost = total_path_cost + coop_cost + way_penalty

    return total_cost, bezier_points_all, way_all


def generate_smooth_path(x_way_uav, threat_interp_num, begin_point, final_point, uav_id):
    """
    生成单架无人机的平滑贝塞尔路径
    """
    x = x_way_uav[::3]
    y = x_way_uav[1::3]
    z = x_way_uav[2::3]

    if isinstance(begin_point, list):
        start_pt = np.array(begin_point[uav_id - 1])
        end_pt = np.array(final_point[uav_id - 1]) if isinstance(final_point, list) else np.array(final_point)
    else:
        start_pt = np.array(begin_point)
        end_pt = np.array(final_point)

    way = np.vstack([start_pt, np.c_[x, y, z], end_pt])
    N = way.shape[0]
    k_bezier = 0.2
    n_samples_per_curve = threat_interp_num
    t_samples = np.linspace(0, 1, n_samples_per_curve)[:, None]
    one_minus_t = 1 - t_samples

    n_curves = N - 1
    control_points = np.zeros((n_curves * 4, 3))

    for i in range(n_curves):
        P0 = way[i]
        P3 = way[i + 1]

        if i == 0:
            dir_vec1 = P3 - P0
            P1 = P0 + k_bezier * dir_vec1
            P2 = P3 - k_bezier * (way[i + 2] - P3) if N >= 3 else P3 - k_bezier * dir_vec1
        elif i < n_curves - 1:
            P1 = 2 * P0 - control_points[4 * (i - 1) + 2]
            P2 = P3 - k_bezier * (way[i + 2] - P3)
        else:
            P1 = 2 * P0 - control_points[4 * (i - 1) + 2]
            P2 = P3 - k_bezier * (P3 - P0)

        control_points[4 * i:4 * (i + 1)] = np.vstack([P0, P1, P2, P3])

    bezier_points = []
    for i in range(n_curves):
        idx = 4 * i
        P0, P1, P2, P3 = control_points[idx:idx + 4]

        Bt = (one_minus_t ** 3 * P0 +
              3 * one_minus_t ** 2 * t_samples * P1 +
              3 * one_minus_t * t_samples ** 2 * P2 +
              t_samples ** 3 * P3)

        bezier_points.append(Bt[:-1] if i < n_curves - 1 else Bt)

    bezier_points = np.vstack(bezier_points)
    return bezier_points, way


def compute_single_path_cost(bezier_points, x_map, y_map, z_map):
    """
    计算单条路径的代价（距离+地形威胁）
    """
    if bezier_points.shape[0] >= 2:
        distance_cost = np.sum(np.linalg.norm(np.diff(bezier_points, axis=0), axis=1))
    else:
        distance_cost = 0.0

    r_safe = 8.0
    r_safe_sq = r_safe ** 2
    collision_penalty = 1e8

    x_grid_all = x_map[0]
    y_grid_all = y_map[:, 0]
    z_max = z_map.max()

    valid_mask = bezier_points[:, 2] <= z_max + r_safe
    valid_points = bezier_points[valid_mask]
    n_valid = valid_points.shape[0]
    if n_valid == 0:
        return distance_cost

    x_min = valid_points[:, 0] - r_safe
    x_max = valid_points[:, 0] + r_safe
    y_min = valid_points[:, 1] - r_safe
    y_max = valid_points[:, 1] + r_safe

    x_start = np.searchsorted(x_grid_all, x_min, side='left').clip(0, len(x_grid_all) - 1)
    x_end = np.searchsorted(x_grid_all, x_max, side='right').clip(0, len(x_grid_all) - 1)
    y_start = np.searchsorted(y_grid_all, y_min, side='left').clip(0, len(y_grid_all) - 1)
    y_end = np.searchsorted(y_grid_all, y_max, side='right').clip(0, len(y_grid_all) - 1)

    threat_cost = 0.0

    for idx in range(n_valid):
        px, py, pz = valid_points[idx]
        xs, xe = x_start[idx], x_end[idx]
        ys, ye = y_start[idx], y_end[idx]

        x_local = x_map[ys:ye + 1, xs:xe + 1]
        y_local = y_map[ys:ye + 1, xs:xe + 1]
        z_local = z_map[ys:ye + 1, xs:xe + 1]

        if z_local.max() >= pz:
            threat_cost += collision_penalty * (1 + z_local.max() - pz)
            continue
        elif z_local.max() <= pz - r_safe:
            continue

        dx = x_local - px
        dy = y_local - py
        dz = z_local - pz
        dist_sq = dx ** 2 + dy ** 2 + dz ** 2

        mask = dist_sq <= r_safe_sq
        if np.any(mask):
            r_ij = np.sqrt(dist_sq[mask]) + 1e-8
            threat_cost += 100 * np.sum((r_safe / r_ij) ** 2)

    return distance_cost + threat_cost


def compute_cooperation_cost(bezier_points_all, uav_num):
    """
    计算多无人机协同避碰代价
    """
    if uav_num < 2:
        return 0.0

    r_safe = 8.0
    coop_cost = 0.0

    trimmed_points = []
    for pts in bezier_points_all:
        if pts.shape[0] >= 3:
            trimmed_points.append(pts[1:-1])
        else:
            trimmed_points.append(np.empty((0, 3)))

    for i in range(uav_num):
        A = trimmed_points[i]
        if A.size == 0:
            continue
        for j in range(i + 1, uav_num):
            B = trimmed_points[j]
            if B.size == 0:
                continue

            dist_matrix = cdist(A, B)
            valid_dist = dist_matrix[(dist_matrix > 0) & (dist_matrix < r_safe)]
            if valid_dist.size > 0:
                coop_cost += 100 * np.sum((r_safe / valid_dist) ** 2)

    return coop_cost