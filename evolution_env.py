import random
import time
import sys
import numpy as np
from typing import List, Tuple
# 导入全局默认值，用于兼容老代码
from config import dim, MaxFEs, initial_pop_size, target_pop_size, counternum

try:
    import cec2017_py.functions as functions
except ImportError:
    print("请安装cec2017_py库: pip install cec2017-py")
    sys.exit(1)


def create_action_space():
    topology_options = [i for i in range(3, 31)]  # 拓扑结构大小：3~30
    theta_options = [i * 0.05 for i in range(16)]  # theta：0~0.75
    phi_options = [0.05 * i for i in range(1, 9)]  # phi：0.05~0.4

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


class Benchmarks:
    def compute(self, x: List[float]) -> float:
        raise NotImplementedError("需实现CEC2017函数的compute方法")

    def getMinX(self) -> float:
        return -100.0

    def getMaxX(self) -> float:
        return 100.0


def generate_func_obj(func_id: int) -> Benchmarks:
    if 1 <= func_id <= 30:
        class F(Benchmarks):
            def __init__(self, func_id: int):
                self.func_id = func_id
                self.cec_func = getattr(functions, f"f{func_id}")
                self.opt_val = 100 * func_id  # CEC2017理论最优值

            def compute(self, x: List[float]) -> float:
                x_np = np.array(x, dtype=np.float64).reshape(1, -1)
                return self.cec_func(x_np)[0]

        return F(func_id)
    else:
        print("无效的函数索引，CEC2017仅支持1-30", file=sys.stderr)
        sys.exit(-1)


class EvolutionaryEnv:
    # 关键：参数默认值全部取全局配置，老代码不传参时和原行为完全一致
    def __init__(self, func_id=1, dim_env=dim, max_fes_env=MaxFEs, init_pop=initial_pop_size, tgt_pop=target_pop_size):
        self.func_id = func_id
        self.dim = dim_env
        self.MaxFEs = max_fes_env
        self.initial_pop_size = init_pop
        self.target_pop_size = tgt_pop

        self.fp = generate_func_obj(func_id)
        self.opt_val = self.fp.opt_val

        self.actions, self.action_map, self.action_dim = create_action_space()
        self.state_dim = 3

        self.step_logs = []
        self.prev_cos_sim_history = []
        self.current_cos_sim_history = []

        # 收敛曲线记录（新增，不影响原有逻辑）
        self.record_points = np.array([int(self.MaxFEs * (i + 1) / counternum) for i in range(counternum)], dtype=int)
        self.convergence_records = np.zeros(counternum + 1, dtype=np.float64)
        self.record_counter = 0

        self.reset()

    def reset(self):
        self.FEs = 0
        self.current_pop_size = self.initial_pop_size
        self.population_index = list(range(self.current_pop_size))

        self.position, self.velocity = self.initialization()

        self.results, self.gl_best_idx, self.best_fitness, fe = self.fitness_computation()
        self.FEs += fe

        self.prev_cos_sim_history = []
        self.current_cos_sim_history = []

        self.prev_results = self.results.copy()
        self.prev_best_fitness = self.best_fitness
        self.max_global_progress_history = 1e-10

        self.step_logs = []

        # 重置收敛曲线
        self.record_counter = 0
        self.convergence_records.fill(0.0)
        self.convergence_records[0] = self.best_fitness - self.opt_val

        return self._get_state()

    def initialization(self) -> Tuple[np.ndarray, np.ndarray]:
        seed = int(time.time()) * random.randint(0, sys.maxsize)
        random.seed(seed)

        min_x = self.fp.getMinX()
        max_x = self.fp.getMaxX()

        position = np.empty((self.current_pop_size, self.dim), dtype=np.float64)
        velocity = np.zeros((self.current_pop_size, self.dim), dtype=np.float64)

        for i in range(self.current_pop_size):
            for j in range(self.dim):
                position[i, j] = random.uniform(min_x, max_x)

        return position, velocity

    def fitness_computation(self) -> Tuple[np.ndarray, int, float, int]:
        results = np.zeros(self.current_pop_size, dtype=np.float64)
        results[0] = self.fp.compute(self.position[0].tolist())
        best = results[0]
        gbest_idx = 0
        FEs = 1

        for i in range(1, self.current_pop_size):
            results[i] = self.fp.compute(self.position[i].tolist())
            if results[i] < best:
                best = results[i]
                gbest_idx = i
            FEs += 1

        return results, gbest_idx, best, FEs

    def _get_state(self) -> np.ndarray:
        fes_ratio = self.FEs / self.MaxFEs

        mean_position = np.mean(self.position, axis=0)
        distances = np.linalg.norm(self.position - mean_position, axis=1)
        diversity = 5 * np.mean(distances) / (np.sqrt(self.dim) * 100.0)
        diversity = np.clip(diversity, 0, 1)

        avg_cos_sim = np.mean(self.prev_cos_sim_history) if self.prev_cos_sim_history else 0.5
        avg_cos_sim = np.clip(avg_cos_sim, 0, 1)

        return np.array([fes_ratio, diversity, avg_cos_sim], dtype=np.float32)

    def _get_current_pop_size(self):
        fes_ratio = self.FEs / self.MaxFEs
        current_size = int(self.initial_pop_size - (self.initial_pop_size - self.target_pop_size) * fes_ratio)
        return max(self.target_pop_size, min(current_size, self.initial_pop_size))

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

        for _ in range(steps):
            if self.FEs >= self.MaxFEs:
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

                    theta = np.random.uniform(low=-theta_max, high=theta_max, size=self.dim)
                    vel_i, pos_i = self.rule2(vel_i, pos_i, pos_ex1, pos_ex2, theta, phi_selected)

                    pos_i = np.clip(pos_i, self.fp.getMinX(), self.fp.getMaxX())
                    self.velocity[i] = vel_i
                    self.position[i] = pos_i

                    self.FEs += 1
                    new_val = self.fp.compute(pos_i.tolist())

                    individual_progress = old_val - new_val
                    reward1_sum += individual_progress
                    individual_progresses.append(individual_progress)

                    self.results[i] = new_val
                    if new_val < self.best_fitness:
                        self.best_fitness = new_val

                    # 记录收敛点（新增，仅额外记录，不影响核心逻辑）
                    while self.record_counter < counternum and self.FEs >= self.record_points[self.record_counter]:
                        self.convergence_records[self.record_counter + 1] = self.best_fitness - self.opt_val
                        self.record_counter += 1

        self.prune_population()
        self.prev_cos_sim_history = self.current_cos_sim_history.copy()

        # 奖励计算（完全和原来一致）
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

        done = self.FEs >= self.MaxFEs
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

        random_vec = np.random.randn(self.dim)
        v_raw1 = random_vec - np.dot(random_vec, u1) * u1
        while np.linalg.norm(v_raw1) < 1e-10:
            random_vec = np.random.randn(self.dim)
            v_raw1 = random_vec - np.dot(random_vec, u1) * u1
        v1 = v_raw1 / np.linalg.norm(v_raw1)

        random_vec = np.random.randn(self.dim)
        v_raw2 = random_vec - np.dot(random_vec, u2) * u2
        while np.linalg.norm(v_raw2) < 1e-10:
            random_vec = np.random.randn(self.dim)
            v_raw2 = random_vec - np.dot(random_vec, u2) * u2
        v2 = v_raw2 / np.linalg.norm(v_raw2)

        step1 = d1 * (np.cos(theta) * u1 + np.sin(theta) * v1)
        step2 = d2 * (np.cos(theta) * u2 + np.sin(theta) * v2)

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        r3 = np.random.rand(self.dim)
        vel_i = r1 * vel_i + r2 * step1 + phi * r3 * step2
        pos_i += vel_i
        return vel_i, pos_i