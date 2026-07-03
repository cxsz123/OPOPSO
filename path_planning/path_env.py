import random
import time
import sys
import numpy as np
from typing import List, Tuple
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from path_core import fitness


def create_action_space():
    topology_options = [i for i in range(3, 31)]
    theta_options = [i * 0.05 for i in range(16)]
    phi_options = [0.05 * i for i in range(1, 9)]

    actions = []
    action_map = {}
    idx = 0
    for topo in topology_options:
        for theta in theta_options:
            for phi in phi_options:
                actions.append((topo, theta, phi))
                action_map[idx] = (topo, theta, phi)
                idx += 1
    return actions, action_map, len(actions)


class NewType:
    __slots__ = ['data', 'id']

    def __init__(self):
        self.data = 0.0
        self.id = 0

    def __repr__(self) -> str:
        return f"NewType(data={self.data}, id={self.id})"


class PathPlanningFunction:

    def __init__(self, hemi_base_params, cylinder_params, env_config):
        mat_data = loadmat(env_config["mat_file"])
        self.land_sub_rot_cropped = mat_data['land_sub_rot_cropped']

        x_grid = np.arange(0, env_config["X_MAX"])
        y_grid = np.arange(0, env_config["Y_MAX"])
        self.x_map, self.y_map = np.meshgrid(x_grid, y_grid, indexing='xy')
        self.z_map = self.land_sub_rot_cropped

        self.cylinder_params = cylinder_params

        if hemi_base_params.size > 0:
            hemi_xy = hemi_base_params[:, :2]
            interp = RegularGridInterpolator((y_grid, x_grid), self.z_map, method='linear')
            hemi_z = interp(hemi_xy[:, [1, 0]])
            self.hemisphere_params = np.hstack([hemi_xy, hemi_z.reshape(-1, 1), hemi_base_params[:, 2:3]])
        else:
            self.hemisphere_params = np.array([])

        self.begin_point = env_config["begin_point"]
        self.final_point = env_config["final_point"]
        self.env_config = env_config
        self.uav_num = None
        self.dim = None

    def compute(self, x_way):
        total_cost, _, _ = fitness(
            x_way,
            threat_interp_num=self.env_config["THREAT_INTERP_NUM"],
            uav_num=self.uav_num,
            begin_point=self.begin_point,
            final_point=self.final_point,
            x_map=self.x_map,
            y_map=self.y_map,
            z_map=self.z_map,
            x_max=self.env_config["X_MAX"],
            y_max=self.env_config["Y_MAX"],
            z_max=self.env_config["Z_MAX"],
            hemisphere_params=self.hemisphere_params,
            cylinder_params=self.cylinder_params
        )
        return total_cost

    def getMinX(self):
        return np.tile([0, 0, self.env_config["min_z"]], self.dim // 3)

    def getMaxX(self):
        return np.tile([self.env_config["X_MAX"], self.env_config["Y_MAX"], self.env_config["Z_MAX"]], self.dim // 3)


class EvolutionaryEnv:
    """路径规划进化环境"""

    def __init__(self, scenario_hemi_base, scenario_cylinder, env_config, uav_num,
                 single_uav_population=None, pop_single_end=None,
                 fes_multi=None, fes_single_fixed=None):
        self.env_config = env_config
        self.fp = PathPlanningFunction(scenario_hemi_base, scenario_cylinder, env_config)
        self.single_uav_population = single_uav_population

        # ===== 修复：在reset之前先设置好无人机数量和维度 =====
        self.fp.uav_num = uav_num
        self.fp.dim = uav_num * env_config["N_PER_UAV"]

        self.pop_single_end = pop_single_end or env_config.get("POP_SINGLE_END", 100)
        self.fes_multi = fes_multi or env_config.get("FES_MULTI", 80000)
        self.fes_single_fixed = fes_single_fixed or env_config.get("FES_SINGLE_FIXED", 20000)

        self.record_count = 0
        self.record_interval = 0
        self.record_thresholds = []
        self.fitness_records = []
        self.next_record_idx = 0

        if self.single_uav_population is not None:
            self.record_count = 10
            self.record_interval = self.fes_multi // 10
            self.record_thresholds = [self.record_interval * i for i in range(self.record_count)]

        self.actions, self.action_map, self.action_dim = create_action_space()
        self.state_dim = 3

        self.step_logs = []
        self.prev_cos_sim_history = []
        self.current_cos_sim_history = []

        self.reset()

    def reset(self):
        self.FEs = 0
        self.current_pop_size = len(self.single_uav_population) if self.single_uav_population is not None else self.env_config["POP_INIT"]
        self.population_index = list(range(self.current_pop_size))

        self.position, self.velocity = self.initialization()
        self.results, self.gl_best_idx, self.best_fitness, fe = self.fitness_computation()
        self.FEs += fe

        self.prev_cos_sim_history = []
        self.current_cos_sim_history = []
        self.prev_results = self.results.copy()
        self.prev_best_fitness = self.best_fitness
        self.max_global_progress_history = 1e-10

        if self.single_uav_population is not None:
            self.fitness_records = []
            self.next_record_idx = 0
            if self.next_record_idx < self.record_count and self.FEs >= self.record_thresholds[self.next_record_idx]:
                self.fitness_records.append(self.best_fitness)
                self.next_record_idx += 1

        return self._get_state()

    def initialization(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.single_uav_population is not None and len(self.single_uav_population) > 0:
            single_pop = self.single_uav_population
            m = len(single_pop)
            n = self.fp.uav_num
            dim_per_uav = self.env_config["N_PER_UAV"]

            position = np.empty((m, n * dim_per_uav), dtype=np.float64)
            velocity = np.zeros((m, n * dim_per_uav), dtype=np.float64)

            for i in range(m):
                base_individual = single_pop[i]
                valid_indices = [idx for idx in range(m) if idx != i]
                if len(valid_indices) >= n - 1:
                    selected_indices = random.sample(valid_indices, k=n - 1)
                else:
                    selected_indices = random.choices(valid_indices, k=n - 1)

                selected_individuals = [single_pop[idx] for idx in selected_indices]
                combined_individual = np.concatenate([base_individual] + selected_individuals)
                position[i] = combined_individual

            return position, velocity
        else:
            seed = int(time.time()) * random.randint(0, sys.maxsize)
            random.seed(seed)

            min_x = self.fp.getMinX()
            max_x = self.fp.getMaxX()
            dim = self.fp.dim

            position = np.empty((self.current_pop_size, dim), dtype=np.float64)
            velocity = np.zeros((self.current_pop_size, dim), dtype=np.float64)

            for i in range(self.current_pop_size):
                for j in range(dim):
                    position[i, j] = random.uniform(min_x[j], max_x[j])

            return position, velocity

    def fitness_computation(self) -> Tuple[np.ndarray, int, float, int]:
        results = np.zeros(self.current_pop_size, dtype=np.float64)
        results[0] = self.fp.compute(self.position[0])
        best = results[0]
        gbest_idx = 0
        FEs = 1

        for i in range(1, self.current_pop_size):
            results[i] = self.fp.compute(self.position[i])
            if results[i] < best:
                best = results[i]
                gbest_idx = i
            FEs += 1

        return results, gbest_idx, best, FEs

    def _get_state(self) -> np.ndarray:
        fes_total = self.fes_single_fixed if self.single_uav_population is None else self.fes_multi
        fes_ratio = self.FEs / fes_total

        mean_position = np.mean(self.position, axis=0)
        distances = np.linalg.norm(self.position - mean_position, axis=1)
        min_x = self.fp.getMinX()
        max_x = self.fp.getMaxX()
        range_x = max_x - min_x
        diversity = 5 * np.mean(distances) / (np.sqrt(self.fp.dim) * np.mean(range_x)/2)
        diversity = np.clip(diversity, 0, 1)

        avg_cos_sim = np.mean(self.prev_cos_sim_history) if self.prev_cos_sim_history else 0.5
        avg_cos_sim = np.clip(avg_cos_sim, 0, 1)

        return np.array([fes_ratio, diversity, avg_cos_sim], dtype=np.float32)

    def _get_current_pop_size(self):
        if self.single_uav_population is None:
            fes_ratio = min(self.FEs / self.fes_single_fixed, 1.0)
            current_size = self.env_config["POP_INIT"] - (self.env_config["POP_INIT"] - self.pop_single_end) * fes_ratio
        else:
            fes_ratio = min(self.FEs / self.fes_multi, 1.0)
            current_size = self.pop_single_end - (self.pop_single_end - self.env_config["POP_FINAL"]) * fes_ratio

        current_size = int(round(current_size))
        current_size = max(self.env_config["POP_FINAL"], min(current_size, self.env_config["POP_INIT"]))
        return current_size

    def prune_population(self):
        target_size = self._get_current_pop_size()
        if self.current_pop_size <= target_size:
            return

        sorted_indices = np.argsort(self.results)
        keep_indices = sorted_indices[:target_size]

        self.position = self.position[keep_indices]
        self.velocity = self.velocity[keep_indices]
        self.results = self.results[keep_indices]
        self.population_index = list(range(target_size))
        self.current_pop_size = target_size

        self.gl_best_idx = np.argmin(self.results)
        self.best_fitness = self.results[self.gl_best_idx]

    def step(self, action_idx) -> Tuple[np.ndarray, float, bool]:
        topology_size_max, theta_max, phi_selected = self.action_map[action_idx]

        current_prev_best = self.prev_best_fitness
        reward1_sum = 0.0
        individual_progresses = []

        steps = 10
        self.current_cos_sim_history.clear()

        fes_total = self.fes_single_fixed if self.single_uav_population is None else self.fes_multi
        dim = self.fp.dim

        for _ in range(steps):
            if self.FEs >= fes_total:
                break

            topology_size = max(2, min(topology_size_max, self.current_pop_size - 1))
            shuffled_index = self.population_index.copy()
            random.shuffle(shuffled_index)
            subswarm = [shuffled_index[i:i + topology_size] for i in range(0, self.current_pop_size, topology_size)]

            for i in range(self.current_pop_size):
                topology = []
                for group in subswarm:
                    if i in group:
                        m = 0
                        for j in range(len(group) - 1):
                            if group[m] != i:
                                topology.append(NewType())
                                topology[j].id = group[m]
                                topology[j].data = self.results[group[m]]
                                m += 1
                topology_sorted = sorted(topology, key=lambda x: x.data)
                if len(topology) < 2:
                    continue

                if topology_sorted[1].data <= self.results[i]:
                    exemplar1 = topology_sorted[0].id

                    j = len(topology) - 1
                    while j >= 0 and self.results[topology_sorted[j].id] > self.results[i]:
                        j -= 1
                    random_number = random.randint(1, j)
                    exemplar2 = topology_sorted[random_number].id

                    vec1 = self.position[exemplar1] - self.position[i]
                    vec2 = self.position[exemplar2] - self.position[i]
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                        normalized_cos_sim = (cos_sim + 1) / 2
                        self.current_cos_sim_history.append(normalized_cos_sim)

                    old_val = self.results[i]
                    pos_i = self.position[i].copy()
                    vel_i = self.velocity[i].copy()
                    pos_ex1 = self.position[exemplar1]
                    pos_ex2 = self.position[exemplar2]

                    theta = np.random.uniform(low=-theta_max, high=theta_max, size=dim)
                    vel_i, pos_i = self.rule2(vel_i, pos_i, pos_ex1, pos_ex2, theta, phi_selected)

                    min_x = self.fp.getMinX()
                    max_x = self.fp.getMaxX()
                    pos_i = np.clip(pos_i, min_x, max_x)

                    self.velocity[i] = vel_i
                    self.position[i] = pos_i

                    self.FEs += 1
                    new_val = self.fp.compute(pos_i)

                    if self.single_uav_population is not None:
                        while self.next_record_idx < self.record_count and self.FEs >= self.record_thresholds[self.next_record_idx]:
                            self.fitness_records.append(self.best_fitness)
                            self.next_record_idx += 1

                    individual_progress = old_val - new_val
                    reward1_sum += individual_progress
                    individual_progresses.append(individual_progress)

                    self.results[i] = new_val
                    if new_val < self.best_fitness:
                        self.best_fitness = new_val

        self.prune_population()
        self.prev_cos_sim_history = self.current_cos_sim_history.copy()

        if individual_progresses:
            max_progress = max(individual_progresses)
            min_progress = min(individual_progresses)
            total_range = max(max_progress - min_progress, 1e-10)
            normalized = (reward1_sum - min_progress) / total_range
            reward1_scaled = 2 * normalized - 1
        else:
            reward1_scaled = 0.0
        reward1_scaled = np.clip(reward1_scaled, -1, 1)

        current_global_progress = max(current_prev_best - self.best_fitness, 0)
        if current_global_progress > self.max_global_progress_history:
            self.max_global_progress_history = current_global_progress
        reward2_scaled = 10 * (current_global_progress / self.max_global_progress_history) if self.max_global_progress_history > 0 else 0
        reward2_scaled = np.clip(reward2_scaled, 0, 10)

        total_reward = reward1_scaled + reward2_scaled

        self.step_logs.append({
            'fes': self.FEs,
            'reward': total_reward,
            'best_fitness': self.best_fitness,
            'action': (topology_size_max, theta_max, phi_selected),
            'current_pop_size': self.current_pop_size
        })

        self.prev_results = self.results.copy()
        self.prev_best_fitness = self.best_fitness

        done = self.FEs >= fes_total
        next_state = self._get_state()

        return next_state, total_reward, done

    def rule2(self, vel_i: np.ndarray, pos_i: np.ndarray, pos_ex1: np.ndarray,
              pos_ex2: np.ndarray, theta: np.ndarray, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        delta1 = pos_ex1 - pos_i
        d1 = np.linalg.norm(delta1)
        delta2 = pos_ex2 - pos_i
        d2 = np.linalg.norm(delta2)

        if d1 < 1e-10 or d2 < 1e-10:
            return vel_i, pos_i

        u1 = delta1 / d1
        u2 = delta2 / d2
        dim = self.fp.dim

        random_vec = np.random.randn(dim)
        v_raw1 = random_vec - np.dot(random_vec, u1) * u1
        while np.linalg.norm(v_raw1) < 1e-10:
            random_vec = np.random.randn(dim)
            v_raw1 = random_vec - np.dot(random_vec, u1) * u1
        v1 = v_raw1 / np.linalg.norm(v_raw1)

        random_vec = np.random.randn(dim)
        v_raw2 = random_vec - np.dot(random_vec, u2) * u2
        while np.linalg.norm(v_raw2) < 1e-10:
            random_vec = np.random.randn(dim)
            v_raw2 = random_vec - np.dot(random_vec, u2) * u2
        v2 = v_raw2 / np.linalg.norm(v_raw2)

        step1 = d1 * (np.cos(theta) * u1 + np.sin(theta) * v1)
        step2 = d2 * (np.cos(theta) * u2 + np.sin(theta) * v2)

        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        r3 = np.random.rand(dim)
        vel_i = r1 * vel_i + r2 * step1 + phi * r3 * step2
        pos_i += vel_i

        return vel_i, pos_i