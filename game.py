import numpy as np
import random
import logic
import constants as c
from collections import Counter
import math

def enhanced_heuristic(state):
    # 空单元格数量
    empty_cells = np.sum(state == 0)
    
    # 平滑性 - 相邻单元格的差值
    smoothness = 0
    for i in range(4):
        for j in range(4):
            if state[i,j] != 0:
                value = math.log2(state[i,j])
                for dx, dy in [(0,1), (1,0)]:
                    if 0 <= i+dx < 4 and 0 <= j+dy < 4 and state[i+dx,j+dy] != 0:
                        smoothness -= abs(value - math.log2(state[i+dx,j+dy]))
    
    # 单调性 - 检查行和列的单调递增/递减
    # monotonicity = 0
    # for i in range(4):
    #     for j in range(3):
    #         if state[i,j] != 0 and state[i,j+1] != 0:
    #             if state[i,j] > state[i,j+1]:
    #                 monotonicity += math.log2(state[i,j]) - math.log2(state[i,j+1])
    #             else:
    #                 monotonicity += math.log2(state[i,j+1]) - math.log2(state[i,j])
    
    # for j in range(4):
    #     for i in range(3):
    #         if state[i,j] != 0 and state[i+1,j] != 0:
    #             if state[i,j] > state[i+1,j]:
    #                 monotonicity += math.log2(state[i,j]) - math.log2(state[i+1,j])
    #             else:
    #                 monotonicity += math.log2(state[i+1,j]) - math.log2(state[i,j])
    
    # 最大数字在角落
    max_tile = np.max(state)
    corner_bonus = 0
    if state[0,0] == max_tile or state[0,3] == max_tile or state[3,0] == max_tile or state[3,3] == max_tile:
        corner_bonus = math.log2(max_tile) * 2

    # 合并潜力
    merge_potential = 0
    for i in range(4):
        for j in range(3):
            if state[i,j] == state[i,j+1] and state[i,j] != 0:
                merge_potential += math.log2(state[i,j])
    
    for j in range(4):
        for i in range(3):
            if state[i,j] == state[i+1,j] and state[i,j] != 0:
                merge_potential += math.log2(state[i,j])
    
    weights = {
        'empty': 10,
        'smoothness': 0.1,
        # 'monotonicity': 1.0,
        'corner': 5,
        'merge': 2
    }
    
    score = (weights['empty'] * empty_cells +
             weights['smoothness'] * smoothness +
            #  weights['monotonicity'] * monotonicity +
             weights['corner'] * corner_bonus +
             weights['merge'] * merge_potential)
    
    return score

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.score = 0
        self.matrix = None
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        # random.seed(42)
        self.score = 0
        self.matrix = logic.new_game(self.size)
        return self.get_state()
    
    def get_state(self):
        """获取当前游戏状态"""
        return np.array(self.matrix)
    
    def get_max_tile(self):
        """获取棋盘上的最大数字"""
        return np.max(self.matrix)
    
    def get_available_actions(self):
        """获取当前可用的动作"""
        available_actions = []
        
        # 检查每个方向是否可以移动
        # 0: 上, 1: 下, 2: 左, 3: 右
        test_matrix = [row[:] for row in self.matrix]  # 深拷贝矩阵
        
        # 上
        new_matrix, done = logic.up(test_matrix)
        if done:
            available_actions.append(1)
        else:
            available_actions.append(0)
        
        # 下
        test_matrix = [row[:] for row in self.matrix]
        new_matrix, done = logic.down(test_matrix)
        if done:
            available_actions.append(1)
        else:
            available_actions.append(0)
        
        # 左
        test_matrix = [row[:] for row in self.matrix]
        new_matrix, done = logic.left(test_matrix)
        if done:
            available_actions.append(1)
        else:
            available_actions.append(0)
        
        # 右
        test_matrix = [row[:] for row in self.matrix]
        new_matrix, done = logic.right(test_matrix)
        if done:
            available_actions.append(1)
        else:
            available_actions.append(0)
        
        return available_actions
    
    def move(self, action):
        """执行动作并返回新状态、奖励、是否结束和额外信息"""
        old_max = self.get_max_tile()
        old_sum = sum(sum(row) for row in self.matrix)
        old_cnt = Counter([x for row in self.matrix for x in row])
        old_score = enhanced_heuristic(self.get_state())

        # 执行动作
        # 0: 上, 1: 下, 2: 左, 3: 右
        if action == 0:
            self.matrix, done = logic.up(self.matrix)
        elif action == 1:
            self.matrix, done = logic.down(self.matrix)
        elif action == 2:
            self.matrix, done = logic.left(self.matrix)
        elif action == 3:
            self.matrix, done = logic.right(self.matrix)
        else:
            raise ValueError(f"无效的动作: {action}")
        
        # 如果移动有效，添加新的数字
        if done:
            self.matrix = logic.add_two(self.matrix)
        
        # 计算奖励
        new_sum = sum(sum(row) for row in self.matrix)
        new_max = self.get_max_tile()
        new_cnt = Counter([x for row in self.matrix for x in row])
        new_score = enhanced_heuristic(self.get_state())

        # 奖励策略：合并得分 + 最大数字提升奖励
        merge_reward = (new_sum - old_sum) / 10.0  # 合并得分
        max_tile_reward = 0
        if new_max > old_max:
            max_tile_reward = np.log2(new_max) - np.log2(old_max)  # 最大数字提升奖励
        
        reward = np.log2(new_max) / 11 + max_tile_reward
        reward += (new_score - old_score) / 30

        # 如果移动无效，给予负奖励
        if not done:
            reward = -1.0
        
        # 检查游戏状态
        game_over = False
        game_state = logic.game_state(self.matrix)
        if game_state == 'win':
            reward += 10.0  # 胜利额外奖励
            game_over = True
        elif game_state == 'lose':
            reward -= 5.0  # 失败惩罚
            game_over = True
        
        # 更新分数
        self.score += int(merge_reward * 10)
        
        return self.get_state(), reward, game_over, {"score": self.score, "max_tile": new_max}