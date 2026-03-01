import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import ast
import torch
import random
from stable_baselines3.common.evaluation import evaluate_policy
from PPO环境构建 import *

data_file_path = f'北京-上海数据集-未购票惩罚.xlsx'
data = pd.read_excel(data_file_path)

# 划分数据集
split_date = pd.Timestamp('2024-03-03 00:00:00')
train_data = data[data['航班出发日期'] < split_date]
test_data = data[data['航班出发日期'] >= split_date]

# 变量
selected_static_features = ['航空公司', '出发机场', '出发机场']
num_future_predictions = 2 # 可调整的机票价格预测值

# 超参数
learning_rate = 0.0003  # 学习率
n_steps = 3  # 更新策略时间步数
gamma = 0.99  # 折扣因子
policy_kwargs = dict(net_arch=[256, 128])
total_timesteps = 500000  # 总训练步数
num_envs = 14
batch_size = n_steps * num_envs  # 批次大小

seed = 42
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
创建环境
"""
def make_env(df, selected_static_features, num_future_predictions):
    def _init():
        env = FlightPriceEnv_Punishment(
            df,
            selected_static_features=selected_static_features,
            num_future_predictions=num_future_predictions)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init

"""
创建训练环境
"""
train_env = DummyVecEnv(
    [make_env(train_data, selected_static_features, num_future_predictions) for _ in range(num_envs)])

model = PPO(
    'MlpPolicy',
    train_env,
    verbose=1,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    gamma=gamma,
    policy_kwargs=policy_kwargs,
    device=device,
    ent_coef=0.1,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    clip_range_vf=0.2
)

"""
回调函数
"""
class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.actions = []

    def _on_step(self) -> bool:
        if self.locals.get('infos') is not None:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    if self.verbose > 0:
                        print(
                            f"Episode {len(self.episode_rewards)}: reward={episode_reward:.2f}, length={episode_length}")

        actions = self.locals.get('actions') # 记录动作
        if actions is not None:
            self.actions.extend(actions)
        return True

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.actions = []

custom_callback = CustomLoggingCallback(verbose=1)

"""
模型训练
"""
model.learn(
    total_timesteps=total_timesteps,
    reset_num_timesteps=False,
    callback=custom_callback
)

action_counts = np.bincount(custom_callback.actions) # 计算购买动作比例
total_actions = len(custom_callback.actions)

if len(action_counts) > 1:
    purchase_ratio = action_counts[1] / total_actions * 100
else:
    purchase_ratio = 0.0
custom_callback.reset()

model_save_path = "ppo_purchase_decision_model.zip"
# print(f"保存模型到 {model_save_path}...")
model.save(model_save_path)

"""
模型测试
"""
model = PPO.load(model_save_path, device=device)

env_p = FlightPriceEnv_Punishment(
    test_data,
    selected_static_features=selected_static_features,
    num_future_predictions=num_future_predictions)

mean_reward, std_reward = evaluate_policy(model, env_p, n_eval_episodes=10, render=False)
print(f"奖励平均值: {mean_reward}, 奖励标准差: {std_reward}")

test_sequences = [group.reset_index(drop=True) for _, group in test_data.groupby('序列ID')] # 测试集

"""
统计每个序列的决策动作
"""
for seq_idx, seq_data in enumerate(test_sequences):
    print(f"正在评估序列 {seq_idx + 1}/{len(test_sequences)}...")

    # 创建测试环境
    env = FlightPriceEnv_Punishment(
        test_data,  # 使用当前序列数据作为环境的输入
        selected_static_features=selected_static_features,
        num_future_predictions=num_future_predictions)

    obs, _ = env.reset(sequence_index=seq_idx)

    done = False
    episode_reward = 0
    episode_actions = []  # 记录每个回合的动作

    while not done:
        """
        模型预测
        """
        action, _states = model.predict(obs, deterministic=True)
        episode_actions.append(action)

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    print(f"序列 {seq_idx + 1} 的决策动作: {episode_actions}")
    print(f'样本长度：{len(seq_data)}')
    print(f'决策位置：{len(episode_actions)}')
    print(f"序列 {seq_idx + 1} 的总奖励: {episode_reward}")
    print('==================================================================================================================')

    decision_actions = np.zeros(len(seq_data), dtype=int) # 创建一个新列“决策动作”，根据每个决策的位置填充
    for idx, action in enumerate(episode_actions):
        if action == 1:
            decision_actions[idx] = 1
    seq_data['决策动作'] = decision_actions

    if seq_idx == 0:
        updated_test_data = seq_data
    else:
        updated_test_data = pd.concat([updated_test_data, seq_data], axis=0)

output_file_path = f'PPO测试结果-未购票惩罚.xlsx'
updated_test_data.to_excel(output_file_path, index=False)

purchases_data = updated_test_data[updated_test_data['决策动作'] == 1].copy()  # 筛选出决策动作为1的行

purchases_data['价格差'] = purchases_data['机票价格'] - purchases_data['历史最低价格']  # 计算机票价格-历史最低价格的差值
cumulative_sum = purchases_data['价格差'].sum()  # 计算差值列的累积和
zero_count = (purchases_data['价格差'] == 0).sum()  # 计算价格差为 0 的数量
right_ratio = np.round(zero_count / len(purchases_data) * 100,1)  # 计算准确率
average_result = cumulative_sum / len(purchases_data) if len(purchases_data) > 0 else 0  # 计算平均值：累积和/样本总数
ratio = np.round(len(purchases_data) / len(set(test_data['序列ID'])) * 100,1)  # 计算执行率

# 输出结果
print(f"样本总数: {len(set(test_data['序列ID']))}")
print(f"决策动作为1的样本数: {len(purchases_data)}")
print(f"价格增加的累积和: {cumulative_sum}")
print(f"平均增加花费: {average_result}")
print(f'准确率：{right_ratio}%')
print(f'执行率为：{ratio}%')



