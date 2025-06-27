from PPO import PPOAgent
from rainbow_DQN import QLearningAgent
import time
from puzzle import GameGrid
from game import Game2048
from visualization import GameVisualizer
import argparse

"""
训练模式:
python main.py --mode train --agent dqn --episodes 2000
python main.py --mode train --agent ppo --episodes 2000

AI游玩模式:
python main.py --mode play --agent dqn
python main.py --mode play --agent ppo

人类游玩模式:
python main.py --mode human
"""

def train_agent(agent_type, episodes=2000):
    """训练代理（不带实时可视化，仅在训练结束时保存结果）"""
    if agent_type == "dqn":
        agent = QLearningAgent()
        print(f"开始训练 DQN 代理，计划训练{episodes}轮...")
    elif agent_type == "ppo":
        agent = PPOAgent()
        print(f"开始训练 PPO 代理，计划训练{episodes}轮...")
    else:
        raise ValueError("未知的代理类型")
    
    scores, max_tiles = agent.train(episodes=episodes)
    
    print(f"训练完成！")
    print(f"最后10轮平均分数: {sum(scores[-10:]) / 10:.2f}")
    print(f"最后10轮平均最大数字: {sum(max_tiles[-10:]) / 10:.2f}")
    return agent

def agent_play(agent_type, trained_agent=None):
    """让训练好的代理自动游玩（带可视化）"""
    if trained_agent is None:
        if agent_type == "dqn":
            agent = QLearningAgent()
        elif agent_type == "ppo":
            agent = PPOAgent()
        else:
            raise ValueError("未知的代理类型")
    else:
        agent = trained_agent
    
    print(f"让训练好的 {agent_type.upper()} 代理自动游玩...")
    
    game = Game2048()
    state = game.reset()
    total_reward = 0
    done = False
    steps = 0

    visualizer = GameVisualizer()
    visualizer.update_grid_cells(state)
    visualizer.update_info(game.score, game.get_max_tile(), steps)
    
    while not done and steps < 1000:  
        available_actions = game.get_available_actions()
        if not available_actions:
            break

        if agent_type == "human":
            action = visualizer.get_human_input()
            q_values = None
        else:
            action, q_values = agent.play_move(game, return_q_values=True)
        
        if action is None:
            break

        if q_values is not None:
            visualizer.update_q_values(q_values)
        
        next_state, reward, done, info = game.move(action)
        total_reward += reward
        state = next_state
        steps += 1

        visualizer.update_grid_cells(state)
        visualizer.update_info(game.score, game.get_max_tile(), steps, action)
        visualizer.update()

        print(f"步骤 {steps}, 动作: {['上', '下', '左', '右'][action]}, 分数: {info['score']}, 最大数字: {info['max_tile']}")

        time.sleep(0.3)
    
    print(f"游戏结束！总步数: {steps}, 总分数: {game.score}, 最大数字: {game.get_max_tile()}")
    visualizer.mainloop()  

def human_play():
    print("人类玩家模式...")
    game_grid = GameGrid()

def parse_args():
    parser = argparse.ArgumentParser(description="2048游戏AI代理训练和游玩")
    
    # 主模式选择
    parser.add_argument("--mode", type=str, choices=["train", "play", "human"], 
                       required=True, help="选择模式: train(训练), play(游玩), human(人类游玩)")
    
    # 代理选择
    parser.add_argument("--agent", type=str, choices=["dqn", "ppo"], 
                       help="选择AI代理: dqn 或 ppo (仅在train和play模式需要)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, 
                       help="训练轮数 (仅在train模式需要)")
    parser.add_argument("--episodes", type=int, default=1000, 
                       help="每轮训练次数 (仅在train模式需要)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "train":
        if args.agent is None:
            raise ValueError("训练模式需要指定--agent参数")
        trained_agent = train_agent(args.agent, args.episodes)
    elif args.mode == "play":
        if args.agent is None:
            raise ValueError("游玩模式需要指定--agent参数")
        agent_play(args.agent)
    elif args.mode == "human":
        human_play()