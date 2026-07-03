import os
import time
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from config import dim_list, counternum, FunRunNum
from d3qn_agent import DuelingDDQNAgent
from evolution_env import EvolutionaryEnv


def test_single_function(func_id, model_path, test_runs, state_dim, action_dim, dim_env, max_fes_env, init_pop, tgt_pop):
    agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    func_test_runs = []
    total_rewards = []
    steps_list = []
    fes_list = []
    final_pop_sizes = []
    all_convergence_curves = []

    for run_idx in range(1, test_runs + 1):
        env = EvolutionaryEnv(
            func_id=func_id,
            dim_env=dim_env,
            max_fes_env=max_fes_env,
            init_pop=init_pop,
            tgt_pop=tgt_pop
        )
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
        all_convergence_curves.append(env.convergence_records.copy())

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
    return func_id, func_test_runs, func_statistics, np.array(all_convergence_curves)


def create_result_directories():
    for dir_name in ['models', 'plots', 'logs', 'excel_result']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


if __name__ == "__main__":
    create_result_directories()

    # 外层循环遍历所有维度
    for dim in dim_list:
        print(f"\n==================== 开始处理维度 {dim}D ====================")
        MaxFEs = 10000 * dim
        initial_pop_size = 1000
        target_pop_size = int(0.1 * initial_pop_size)

        # 临时环境获取维度信息
        temp_env = EvolutionaryEnv(
            func_id=1,
            dim_env=dim,
            max_fes_env=MaxFEs,
            init_pop=initial_pop_size,
            tgt_pop=target_pop_size
        )
        state_dim = temp_env.state_dim
        action_dim = temp_env.action_dim
        del temp_env

        final_model_path = f"models/finally.pth"
        TEST_RUNS_PER_FUNC = 1
        NUM_PROCESSES = 10

        print(f"\n===== {dim}D 开始并行测试（{NUM_PROCESSES}个CPU，每个函数运行 {TEST_RUNS_PER_FUNC} 次） =====")
        test_task = partial(
            test_single_function,
            model_path=final_model_path,
            test_runs=TEST_RUNS_PER_FUNC,
            state_dim=state_dim,
            action_dim=action_dim,
            dim_env=dim,
            max_fes_env=MaxFEs,
            init_pop=initial_pop_size,
            tgt_pop=target_pop_size
        )

        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            parallel_results = pool.map(test_task, sorted(FunRunNum))

        test_results = {}
        func_statistics = {}
        conv_curve_dict = {}
        for res in parallel_results:
            func_id, runs, stats, conv_arr = res
            test_results[func_id] = runs
            func_statistics[func_id] = stats
            conv_curve_dict[func_id] = conv_arr

        # 导出Excel
        step_pct = 100 / counternum
        headers = ["初始化"] + [f"{step_pct * i:.0f}%FES" for i in range(1, counternum)] + ["100%FES"]
        xlsx_save_path = f"excel_result/D3QN_OPOPSO_dim{dim}_test_results.xlsx"

        with pd.ExcelWriter(xlsx_save_path, engine="xlsxwriter") as writer:
            summary_rows = [
                ["维度dim", dim],
                ["初始种群", initial_pop_size],
                ["目标种群", target_pop_size],
                ["单函数重复运行次数", TEST_RUNS_PER_FUNC],
                ["总MaxFEs", MaxFEs],
                []
            ]
            summary_rows.append(["函数ID", "最终误差均值", "最终误差标准差"])
            for fid in sorted(FunRunNum):
                stat = func_statistics[fid]["best_fitness"]
                mean_err = stat["mean"] - (100 * fid)
                std_err = stat["std"]
                summary_rows.append([fid, mean_err, std_err])
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="汇总信息", index=False, header=False)

            for fid in sorted(FunRunNum):
                conv_array = conv_curve_dict[fid]
                df_func = pd.DataFrame(
                    conv_array,
                    index=[f"第{i}次运行" for i in range(1, TEST_RUNS_PER_FUNC + 1)],
                    columns=headers
                )
                final_col = conv_array[:, -1]
                df_func.loc["误差均值"] = [np.mean(final_col)] + [np.nan] * counternum
                df_func.loc["误差标准差"] = [np.std(final_col)] + [np.nan] * counternum
                df_func.to_excel(writer, sheet_name=f"F{fid}")

        print(f"\n{dim}D Excel收敛曲线结果已保存至: {xlsx_save_path}")
        print(f"===== {dim}D 并行测试完成 =====")

    print("\n==================== 所有维度全部执行完毕 ====================")