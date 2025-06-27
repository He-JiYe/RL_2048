# RL_2048 
这是一个简单的强化学习入门项目，基于Pytorch框架实现了DQN和PPO的2048游戏项目。本项目的游戏本体来源于项目 https://github.com/evenboos/2048-rainbowDQN.git

# 安装
1. 克隆项目到本地：
```bash
git clone 
cd RL2048
```

2. 安装依赖：
```bash
pip install torch numpy tkinter matplotlib
```

# 使用方法
1. 训练模式:
```bash
python main.py --mode train --agent dqn --episodes 2000
python main.py --mode train --agent ppo --episodes 2000
```

2. AI游玩模式:
```bash
python main.py --mode play --agent dqn
python main.py --mode play --agent ppo
```

3. 人类游玩模式:
```bash
python main.py --mode human
```

4. 查看指令帮助:
```bash
python main.py --help

"""
usage: main.py [-h] --mode {train,play,human} [--agent {dqn,ppo}] [--epochs EPOCHS] [--episodes EPISODES]

2048游戏AI代理训练和游玩

options:
  -h, --help            show this help message and exit
  --mode {train,play,human}
                        选择模式: train(训练), play(游玩), human(人类游玩)
  --agent {dqn,ppo}     选择AI代理: dqn 或 ppo (仅在train和play模式需要)
  --epochs EPOCHS       训练轮数 (仅在train模式需要)
  --episodes EPISODES   每轮训练次数 (仅在train模式需要)
"""
```

# 项目结构
- `main.py`: 主程序入口，处理命令行参数和游戏模式选择
- `game.py`: 2048游戏核心逻辑
- `visualization.py`: 游戏界面和训练过程可视化
- `constants.py`: 常量定义
- `logic.py`: 游戏逻辑辅助函数
- `puzzle.py`: 人类玩家界面实现
- `rainbow_DQN.py`: rainbow_DQN代理实现
- `PPO.py`: PPO代理实现
- `utils.py`: DQN和PPO所需要的神经网络和相关函数部分

# 强化学习
__Rainbow DQN:__
在普通DQN的基础上实现了: 
1. 优先级经验回放
2. Double DQN
3. Dueling DQN
4. Distribution DQN
5. n-step trace
6. Noisy Net

__PPO:__
使用masked PPO，重要性采样时记得考虑mask后的动作分布，开始因为忽略了这点导致调了很久。

__奖励设置:__
1. 对状态作对数化预处理
2. 获得当前最大数字正比的奖励(奖励随着局面难度增加而增加)
3. 最大数字提升时，获得奖励数字的提升幅度

__测试结果:__
PPO训练更快，且表现优于DQN，不过都只学到了一个更稳定的策略，最大只能达到1024。


