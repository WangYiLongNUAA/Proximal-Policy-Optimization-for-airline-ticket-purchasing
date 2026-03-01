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

class FlightPriceEnv_Replace(gym.Env):
    def __init__(self, df,selected_static_features, num_future_predictions):
        super(FlightPriceEnv_Replace, self).__init__()

        self.selected_static_features = selected_static_features
        self.num_future_predictions = num_future_predictions
        self.series_id = '序列ID'  # 序列ID用于区分样本
        self.time_index = '时间序列索引'  # 时间序列索引反映时间步顺序
        self.selected_continuous_features = ['机票价格_标准化', '距离航班出发日期天数_标准化','预测值_标准化']  # 连续变量

        self.decision_objective = '替换航班最优价格_标准化'
        self.decision_basis = '原目标航班最低价格_标准化'

        relevant_columns = ([self.series_id,self.time_index,self.decision_objective,self.decision_basis]+
                            self.selected_continuous_features+self.selected_static_features)
        df = df[relevant_columns]

        """
        分类变量独热编码
        """
        categorical_features = self.selected_static_features
        df = pd.get_dummies(df, columns=categorical_features)

        """
        根据序列ID和时间序列索引排序
        """
        df = df.sort_values(by=[self.series_id,self.time_index]).reset_index(drop=True)

        self.sequences = [group.reset_index(drop=True) for _, group in df.groupby(self.series_id)]
        self.num_sequences = len(self.sequences)

        """
        静态变量与连续变量维度
        """
        n_static_features = df.drop(columns=[self.series_id,self.time_index,self.decision_objective,self.decision_basis]
                                            +self.selected_continuous_features).shape[1]
        n_dynamic_features = len(self.selected_continuous_features) + self.num_future_predictions

        """
        状态空间设置
        """
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_static_features + n_dynamic_features,), dtype=np.float32)

        """
        动作空间：0（等待），1（购买）
        """
        self.action_space = spaces.Discrete(2)

        self.current_sequence = None
        self.current_time_index = 0
        self.sequence_length = 0

        self.purchase_executed = False  # 记录购买动作是否已执行

    def reset(self, sequence_index=None, seed=None, options=None):
        super().reset(seed=seed)
        if sequence_index is not None:
            assert 0 <= sequence_index < self.num_sequences, "Invalid sequence index!"
            self.current_sequence_index = sequence_index
        else:
            # 随机选择一个序列ID
            self.current_sequence_index = np.random.randint(self.num_sequences)

        self.current_sequence = self.sequences[self.current_sequence_index]
        self.current_time_index = 0  # 从序列的第一个时间步开始
        self.sequence_length = len(self.current_sequence)

        self.purchase_executed = False  # 重置购买动作执行状态

        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        relevant_data = self.current_sequence.iloc[self.current_time_index]

        static_features = relevant_data.drop([self.series_id,self.time_index,self.decision_objective,self.decision_basis]+self.selected_continuous_features).values

        """
        future_predictions长度的机票价格预测值
        """
        future_predictions = relevant_data['预测值_标准化']

        if isinstance(future_predictions, str):
            future_predictions = ast.literal_eval(future_predictions)
        future_predictions = np.array(future_predictions[:self.num_future_predictions], dtype=np.float32)

        dynamic_features = np.concatenate((
            [relevant_data['机票价格_标准化']],
            [relevant_data['距离航班出发日期天数_标准化']],
            [relevant_data[self.decision_objective]]    ,
            future_predictions
        ), dtype=np.float32)

        """
        合并特征
        """
        observation = np.concatenate([static_features, dynamic_features]).astype(np.float32)

        return observation

    def step(self, action):
        """
        action: 0（等待）或 1（购买）
        返回:
            observation: 新的观察
            reward: 奖励
            done: 是否结束
            truncated: 是否被截断
            info: 额外信息
        """
        relevant_data = self.current_sequence.iloc[self.current_time_index]

        """
        提取动作值
        0（等待）/ 1（购买）
        """
        purchase_action = action

        """
        确保购买动作只能执行一次
        """
        if self.purchase_executed:
            purchase_action = 0

        """
        计算奖励
        """
        reward = self.calculate_reward(relevant_data, purchase_action)

        """
        更新购买动作执行状态
        """
        if purchase_action == 1 and not self.purchase_executed:
            self.purchase_executed = True
            done = True  # 购买后终止序列
        else:
            # 等待
            reward -= 0
            self.current_time_index += 1
            if self.current_time_index >= self.sequence_length: # 检查是否到达序列末尾
                done = True
                if not self.purchase_executed:
                    reward -= 10 # 如果未执行购买动作，给予额外惩罚
            else:
                done = False

        if not done:
            # 获取新的观测
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        truncated = False
        info = {}
        return observation, reward, done, truncated, info

    def calculate_reward(self, row, purchase_action):
        lowest_price = row[self.decision_basis]
        reward = 0

        if purchase_action == 1 and not self.purchase_executed:
            ticket_price = row[self.decision_objective]
            reward_purchase = lowest_price - ticket_price # 计算奖励
            reward += reward_purchase

        return reward


class FlightPriceEnv_Punishment(gym.Env):
    def __init__(self, df, selected_static_features, num_future_predictions):
        super(FlightPriceEnv_Punishment, self).__init__()

        self.selected_static_features = selected_static_features
        self.num_future_predictions = num_future_predictions

        # 保留必要的列
        relevant_columns = ['序列ID', '时间序列索引', '机票价格_标准化', '距离航班出发日期天数_标准化','预测值_标准化',
                            '最低价格_标准化'] + self.selected_static_features

        df = df[relevant_columns]

        # 对分类变量进行独热编码
        categorical_features = self.selected_static_features
        df = pd.get_dummies(df, columns=categorical_features)

        # 根据“序列ID”和“时间序列索引”排序
        df = df.sort_values(by=['序列ID', '时间序列索引']).reset_index(drop=True)

        # 按“序列ID”分组数据
        self.sequences = [group.reset_index(drop=True) for _, group in df.groupby('序列ID')]
        self.num_sequences = len(self.sequences)

        columns_to_exclude = ['序列ID', '时间序列索引', '机票价格_标准化','预测值_标准化',
                              '距离航班出发日期天数_标准化', '最低价格_标准化']

        n_static_features = df.drop(columns=columns_to_exclude).shape[1]
        n_dynamic_features = 2 + self.num_future_predictions  # 当前价格+距离航班出发日期天数+未来预测值，


        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_static_features + n_dynamic_features,), dtype=np.float32)

        # 动作空间，购买动作：0（等待），1（购买）
        self.action_space = spaces.Discrete(2)

        self.current_sequence = None
        self.current_time_index = 0
        self.sequence_length = 0

        # 记录购买动作是否已执行
        self.purchase_executed = False

    def reset(self, sequence_index=None, seed=None, options=None):
        """
        重置环境状态。可以指定序列索引进行测试。

        参数:
            sequence_index (int, optional): 要测试的序列索引

        返回:
            observation (ndarray): 初始观察
            info (dict): 额外信息
        """
        super().reset(seed=seed)

        if sequence_index is not None:
            assert 0 <= sequence_index < self.num_sequences, "Invalid sequence index!"
            self.current_sequence_index = sequence_index
        else:
            # 随机选择一个序列ID
            self.current_sequence_index = np.random.randint(self.num_sequences)

        self.current_sequence = self.sequences[self.current_sequence_index]
        self.current_time_index = 0  # 从序列的第一个时间步开始
        self.sequence_length = len(self.current_sequence)

        # 重置购买动作执行状态
        self.purchase_executed = False

        observation = self._get_observation()  # 获取初始观察
        return observation, {}

    def _get_observation(self):
        """
        获取当前时间步的观察值。

        返回:
            observation (ndarray): 当前观察值
        """
        relevant_data = self.current_sequence.iloc[self.current_time_index]

        static_features = relevant_data.drop(['序列ID', '时间序列索引', '机票价格_标准化','预测值_标准化',
                                              '距离航班出发日期天数_标准化', '最低价格_标准化']).values
        # 动态特征
        # 提取指定数量的未来预测值
        future_predictions = relevant_data['预测值_标准化']

        if isinstance(future_predictions, str):
            future_predictions = ast.literal_eval(future_predictions)

        # 确保提取的预测值数量不超过实际长度
        future_predictions = np.array(future_predictions[:self.num_future_predictions], dtype=np.float32)

        dynamic_features = np.concatenate((
            [relevant_data['机票价格_标准化']],
            future_predictions,
            [relevant_data['距离航班出发日期天数_标准化']]
        ), dtype=np.float32)

        # 合并静态特征和动态特征
        observation = np.concatenate([static_features, dynamic_features]).astype(np.float32)

        return observation

    def step(self, action):
        """
        执行动作，并返回新的观察、奖励、done标志和额外信息。

        参数:
            action (int): 动作，0（等待）或 1（购买）

        返回:
            observation (ndarray): 新的观察
            reward (float): 奖励
            done (bool): 是否结束
            truncated (bool): 是否被截断
            info (dict): 额外信息
        """
        relevant_data = self.current_sequence.iloc[self.current_time_index]

        # 提取动作值
        purchase_action = action  # 动作为整数 0（等待）或 1（购买）

        # 确保购买动作只能执行一次
        if self.purchase_executed:
            purchase_action = 0  # 已经执行过购买动作，不再允许执行

        # 计算奖励
        reward = self.calculate_reward(relevant_data, purchase_action)

        # 更新购买动作执行状态
        if purchase_action == 1 and not self.purchase_executed:
            self.purchase_executed = True
            done = True  # 购买后终止序列
        else:
            # 等待动作
            reward -= 0
            self.current_time_index += 1
            # 检查是否到达序列末尾
            if self.current_time_index >= self.sequence_length:
                done = True
                # 如果未执行购买动作，给予额外惩罚
                if not self.purchase_executed:
                    reward -= 25  # 未购买的惩罚
            else:
                done = False

        if not done:
            # 获取新的观测
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        truncated = False

        # 返回新的观测，奖励，done标志和额外信息
        info = {}
        return observation, reward, done, truncated, info

    def calculate_reward(self, row, purchase_action):
        """
        计算购买决策的奖励。

        参数:
            row (Series): 当前行数据
            purchase_action (int): 动作，0（等待）或 1（购买）

        返回:
            reward (float): 计算得到的奖励
        """
        lowest_price = row['最低价格_标准化']
        reward = 0

        if purchase_action == 1 and not self.purchase_executed:
            ticket_price = row['机票价格_标准化']
            # 计算购买奖励，可以根据需要调整奖励函数
            reward_purchase = lowest_price - ticket_price
            reward += reward_purchase

        return reward
