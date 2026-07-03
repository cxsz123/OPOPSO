import os
import random
import time
import numpy as np
import multiprocessing
from functools import partial
from config import dim, initial_pop_size, target_pop_size, FunRunNum
from d3qn_agent import DuelingDDQNAgent
from evolution_env import EvolutionaryEnv


def train_agent_on_function(env, agent, func_id=1):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action_idx = agent.select_action(state)
        next_state, reward, done = env.step(action_idx)

        agent.memory.add(state, action_idx, reward, next_state, done)
        agent.update_model()

        total_reward += reward
        state = next_state
        steps += 1

        if done:
            break

    return total_reward, env.best_fitness, steps


# 并行测试的单函数任务封装
def test_single_function(func_id, model_path, test_runs, state_dim, action_dim):
    """单个函数的测试任务（供多进程调用）"""
    agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    func_test_runs = []
    total_rewards = []
    steps_list = []
    fes_list = []
    final_pop_sizes = []

    for run_idx in range(1, test_runs + 1):
        env = EvolutionaryEnv(func_id=func_id)
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action_idx = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action_idx)
            total_reward += reward
            state = next_state
            steps += 1

        single_run_result = {
            "best_fitness": env.best_fitness,
            "total_reward": total_reward,
            "steps": steps,
            "total_fes": env.FEs,
            "final_pop_size": env.current_pop_size
        }

        func_test_runs.append(single_run_result)
        total_rewards.append(total_reward)
        steps_list.append(steps)
        fes_list.append(env.FEs)
        final_pop_sizes.append(env.current_pop_size)

    # 计算单个函数的统计信息
    best_fitnesses = [r["best_fitness"] for r in func_test_runs]
    func_statistics = {
        "best_fitness": {
            "mean": np.mean(best_fitnesses),
            "std": np.std(best_fitnesses),
            "min": np.min(best_fitnesses),
            "max": np.max(best_fitnesses),
            "median": np.median(best_fitnesses)
        },
        "total_reward": {
            "mean": np.mean(total_rewards),
            "std": np.std(total_rewards),
            "min": np.min(total_rewards),
            "max": np.max(total_rewards)
        },
        "steps": {
            "mean": np.mean(steps_list),
            "std": np.std(steps_list),
            "min": np.min(steps_list),
            "max": np.max(steps_list)
        },
        "total_fes": {
            "mean": np.mean(fes_list),
            "std": np.std(fes_list),
            "min": np.min(fes_list),
            "max": np.max(fes_list)
        },
        "final_pop_size": {
            "mean": np.mean(final_pop_sizes),
            "std": np.std(final_pop_sizes),
            "min": np.min(final_pop_sizes),
            "max": np.max(final_pop_sizes)
        }
    }

    return func_id, func_test_runs, func_statistics


def create_result_directories():
    for dir_name in ['models', 'plots', 'logs']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


if __name__ == "__main__":
    create_result_directories()

    # 训练阶段：动作维度自动适配扩展后的空间
    all_results = {func_id: {'rewards': [], 'best_fitnesses': [], 'steps': []} for func_id in FunRunNum}
    temp_env = EvolutionaryEnv(func_id=1)
    state_dim = temp_env.state_dim
    action_dim = temp_env.action_dim
    del temp_env

    agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim)
    all_training_rewards = []
    total_episodes = 0
    num_rounds = 120

    for round_num in range(1, num_rounds + 1):
        shuffled_functions = FunRunNum.copy()
        random.shuffle(shuffled_functions)
        round_rewards = []
        round_steps = []

        for func_id in shuffled_functions:
            env = EvolutionaryEnv(func_id=func_id)
            reward, best_fitness, steps = train_agent_on_function(env, agent, func_id=func_id)

            round_rewards.append(reward)
            round_steps.append(steps)
            all_results[func_id]['rewards'].append(reward)
            all_results[func_id]['best_fitnesses'].append(best_fitness)
            all_results[func_id]['steps'].append(steps)
            all_training_rewards.append(reward)

            print(f"函数 {func_id} 训练奖励: {reward:.2f}, 最终适应度: {best_fitness:.6f}")
            agent.update_epsilon()
            total_episodes += 1

        avg_round_reward = np.mean(round_rewards)
        avg_round_steps = np.mean(round_steps)
        print(f"本轮平均奖励: {avg_round_reward:.2f}")

    # 保存模型
    final_model_path = f"models/your_model.pth"
    agent.save(final_model_path)

    # 测试阶段（并行计算）
    TEST_RUNS_PER_FUNC = 35
    NUM_PROCESSES = 10  # 并行CPU数量

    # 准备并行任务的固定参数
    test_task = partial(
        test_single_function,
        model_path=final_model_path,
        test_runs=TEST_RUNS_PER_FUNC,
        state_dim=state_dim,
        action_dim=action_dim
    )

    # 创建进程池，并行处理所有函数
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        parallel_results = pool.map(test_task, sorted(FunRunNum))

    # 整理并行结果
    test_results = {}
    func_statistics = {}
    for res in parallel_results:
        func_id, runs, stats = res
        test_results[func_id] = runs
        func_statistics[func_id] = stats

    # 保存测试日志
    log_path = f"logs/test_results_with_dim{dim}_pop{initial_pop_size}-{target_pop_size}_parallel8.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("===== D3QN智能体测试结果汇总（动态种群900→100+8CPU并行） =====\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"测试函数: {FunRunNum}（共 {len(FunRunNum)} 个，排除函数2）\n")
        f.write(f"每个函数测试次数: {TEST_RUNS_PER_FUNC} 次\n")
        f.write(f"并行CPU数量: {NUM_PROCESSES}\n")
        f.write(f"使用模型: {final_model_path}\n")
        f.write(f"种群配置: 初始{initial_pop_size}→目标{target_pop_size}（线性递减）\n")
        f.write(f"phi取值范围: {[0.05 * i for i in range(1, 9)]}\n\n")

        for func_id in sorted(FunRunNum):
            f.write(f"=====================================\n")
            f.write(f"函数 {func_id} 测试结果\n")
            f.write(f"=====================================\n")

            f.write(f"\n1. 详细测试数据（{TEST_RUNS_PER_FUNC} 次）:\n")
            for run_idx, run_res in enumerate(test_results[func_id], 1):
                f.write(f"   第 {run_idx} 次: ")
                f.write(f"最佳适应度={run_res['best_fitness']:.6f}, ")
                f.write(f"总奖励={run_res['total_reward']:.2f}, ")
                f.write(f"最终种群规模={run_res['final_pop_size']}\n")

            stats = func_statistics[func_id]
            f.write(f"\n2. 统计汇总:\n")
            f.write(f"   最佳适应度: 均值={stats['best_fitness']['mean']:.6f}, ")
            f.write(f"标准差={stats['best_fitness']['std']:.6f}, ")
            f.write(f"最小={stats['best_fitness']['min']:.6f}, ")
            f.write(f"最大={stats['best_fitness']['max']:.6f}, ")
            f.write(f"中位数={stats['best_fitness']['median']:.6f}\n")

    # 打印统计汇总
    print("\n===== 测试统计汇总 =====")
    print(f"{'函数ID':<6} {'平均最佳适应度':<18} {'适应度标准差':<18} {'最小适应度':<18} {'最大适应度':<18}")
    print("-" * 90)
    for func_id in sorted(FunRunNum):
        stats = func_statistics[func_id]
        print(f"{func_id:<6} "
              f"{stats['best_fitness']['mean']:<18.6f} "
              f"{stats['best_fitness']['std']:<18.6f} "
              f"{stats['best_fitness']['min']:<18.6f} "
              f"{stats['best_fitness']['max']:<18.6f}")