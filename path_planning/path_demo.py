import os
import random
import time
import sys
import numpy as np
import multiprocessing
from d3qn_agent import DuelingDDQNAgent
from path_env import EvolutionaryEnv, create_action_space
from path_core import fitness

# ===================== 基础配置 =====================
MODEL_PATH = "models/finally.pth"
TEST_RUNS = 5          # Demo
NUM_PROCESSES = 8      # 并行进程数

UAV_NUM = 2            # 无人机数量
N_PER_UAV = 36         # 单无人机维度
THREAT_INTERP_NUM = 10

# 种群与评估次数参数
POP_INIT = 1000
POP_FINAL = 100
FES_SINGLE_FIXED_BASE = 10000
FES_MULTI_BASE = 40000

# ===================== 场景配置（仅场景1 无障碍） =====================
scene_config = {
    "name": "scene_1_no_obstacle",
    "mat_file": "scene_1.mat",
    "X_MAX": 2001,
    "Y_MAX": 501,
    "Z_MAX": 450,
    "begin_point": np.array([1800, 200, 260]),
    "final_point": np.array([80, 140, 280]),
    "min_z": 250,
    "result_save_dir": "demo_result",
    "N_PER_UAV": N_PER_UAV,
    "THREAT_INTERP_NUM": THREAT_INTERP_NUM,
    "POP_INIT": POP_INIT,
    "POP_FINAL": POP_FINAL
}

# 无障碍场景参数
scenario = {
    "name": "no_obstacles",
    "hemi_base": np.array([]),
    "cylinder": np.array([])
}


# ===================== 单无人机预优化 =====================
def pre_train_single_uav(agent, env_config):
    fes_single_fixed = FES_SINGLE_FIXED_BASE * UAV_NUM
    fes_multi = FES_MULTI_BASE * UAV_NUM
    fes_total = fes_single_fixed + fes_multi
    pop_single_end = int(round(POP_INIT - (POP_INIT - POP_FINAL) * (fes_single_fixed / fes_total)))

    env = EvolutionaryEnv(
        scenario["hemi_base"], scenario["cylinder"], env_config,
        uav_num=1,
        single_uav_population=None,
        pop_single_end=pop_single_end,
        fes_multi=fes_multi,
        fes_single_fixed=fes_single_fixed
    )

    state = env.reset()
    done = False
    while not done and env.FEs < fes_single_fixed:
        action_idx = agent.select_action(state, training=False)
        state, _, done = env.step(action_idx)

    # 对齐最终种群规模
    if abs(env.current_pop_size - pop_single_end) > 1:
        sorted_idx = np.argsort(env.results)[:pop_single_end]
        env.position = env.position[sorted_idx]
        env.results = env.results[sorted_idx]
        env.current_pop_size = pop_single_end
        env.gl_best_idx = np.argmin(env.results)
        env.best_fitness = env.results[env.gl_best_idx]

    return env.position, pop_single_end, fes_multi, fes_single_fixed


# ===================== 单轮测试 =====================
def test_single_run(run_idx, agent, env_config):
    single_uav_pop, pop_single_end, fes_multi, fes_single_fixed = pre_train_single_uav(agent, env_config)

    env = EvolutionaryEnv(
        scenario["hemi_base"], scenario["cylinder"], env_config,
        uav_num=UAV_NUM,
        single_uav_population=single_uav_pop,
        pop_single_end=pop_single_end,
        fes_multi=fes_multi,
        fes_single_fixed=fes_single_fixed
    )

    state = env.reset()
    done = False
    steps = 0
    while not done:
        action_idx = agent.select_action(state, training=False)
        state, _, done = env.step(action_idx)
        steps += 1

    # 补全记录点
    fitness_records = env.fitness_records
    while len(fitness_records) < env.record_count:
        fitness_records.append(env.best_fitness)
    fitness_records.append(env.best_fitness)

    # 提取最优路径详情
    best_x = env.position[env.gl_best_idx]
    fp = env.fp
    total_cost, bezier_points_all, way_all = fitness(
        best_x, THREAT_INTERP_NUM, UAV_NUM,
        fp.begin_point, fp.final_point,
        fp.x_map, fp.y_map, fp.z_map,
        env_config["X_MAX"], env_config["Y_MAX"], env_config["Z_MAX"],
        fp.hemisphere_params, fp.cylinder_params
    )

    print(f"No. {run_idx+1}/{TEST_RUNS} done | cost: {env.best_fitness:.2f}")
    return {
        "run_idx": run_idx + 1,
        "best_fitness": env.best_fitness,
        "steps": steps,
        "fes": env.FEs,
        "best_x": best_x.tolist(),
        "bezier_points": [p.tolist() for p in bezier_points_all],
        "waypoints": [w.tolist() for w in way_all],
        "fitness_records": fitness_records
    }


# ===================== 结果保存 =====================
def save_results(all_results, env_config):
    save_dir = env_config["result_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 统计结果
    fitness_list = [r["best_fitness"] for r in all_results]
    best_idx = np.argmin(fitness_list)
    best_run = all_results[best_idx]

    # 保存汇总
    summary_path = os.path.join(save_dir, "demo_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== 路径规划 Demo 汇总 =====\n")
        f.write(f"场景: {env_config['name']} | 无人机数量: {UAV_NUM}\n")
        f.write(f"测试轮数: {TEST_RUNS}\n\n")
        f.write(f"最优代价: {np.min(fitness_list):.2f}\n")
        f.write(f"平均代价: {np.mean(fitness_list):.2f}\n")
        f.write(f"标准差: {np.std(fitness_list):.2f}\n")
        f.write(f"中位数: {np.median(fitness_list):.2f}\n")

    # 保存最优路径
    best_path = os.path.join(save_dir, "best_path.txt")
    with open(best_path, "w", encoding="utf-8") as f:
        f.write("===== 最优路径详情 =====\n")
        f.write(f"对应轮次: 第 {best_run['run_idx']} 轮 | 代价: {best_run['best_fitness']:.2f}\n\n")
        for i, wp in enumerate(best_run["waypoints"]):
            f.write(f"--- 第 {i+1} 架无人机路径点 ---\n")
            for j, p in enumerate(wp):
                f.write(f"  点{j}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})\n")
            f.write("\n")

    print(f"\n结果已保存至: {save_dir}")
    print(f"  汇总文件: {summary_path}")
    print(f"  最优路径: {best_path}")


# ===================== 主流程 =====================
if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        print("===== Muti-UAVs Demo run =====")
        print(f"场景: {scene_config['name']} | 无人机数: {UAV_NUM} | 测试轮数: {TEST_RUNS}")

        # 加载模型
        state_dim = 3
        _, _, action_dim = create_action_space()
        agent = DuelingDDQNAgent(state_dim, action_dim)
        agent.load(MODEL_PATH)

        # 并行执行测试
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            tasks = [(i, agent, scene_config) for i in range(TEST_RUNS)]
            all_results = pool.starmap(test_single_run, tasks)

        # 保存结果
        save_results(all_results, scene_config)

        # 总耗时
        total = time.perf_counter() - start_time
        print(f"\n全部完成！总耗时: {total:.2f} 秒 ({total/60:.2f} 分钟)")

    except Exception as e:
        print(f"\n运行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()