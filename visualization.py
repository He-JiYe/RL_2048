import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Frame, Label, CENTER
import constants as c
import time
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
try:
    # 尝试设置微软雅黑字体（Windows系统常见字体）
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
    chinese_font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已加载中文字体支持")
except:
    print("无法加载中文字体，将使用默认字体")
    chinese_font = FontProperties()

# 设置matplotlib后端，使用TkAgg以支持交互式显示
matplotlib.use('TkAgg')

class TrainingVisualizer:
    """训练过程可视化类 - 仅收集数据，不进行实时显示"""
    def __init__(self):
        # 初始化数据存储
        self.episodes = []
        self.scores = []
        self.max_tiles = []
        self.losses = []
        self.grads = []
        self.rewards = []
        
        # 创建图表对象，但不显示
        self.fig = None
        self.axes = None
    
    def update(self, episode, score, max_tile, loss=None, grad=None, reward=None):
        """仅更新数据，不进行可视化"""
        self.episodes.append(episode)
        self.scores.append(score)
        self.max_tiles.append(max_tile)
        
        if loss is not None:
            self.losses.append(loss)
        if grad is not None:
            self.grads.append(grad)
        if reward is not None:
            self.rewards.append(reward)
    
    def save_figure(self, filename='training_results.png'):
        """训练结束时生成并保存图表"""
        # 创建图表
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        
        # 尝试设置中文字体
        try:
            font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
            chinese_font = FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            chinese_font = FontProperties()
        
        # 创建图表
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # 设置子图标题 - 使用中文字体
        self.axes[0, 0].set_title('平均分数', fontproperties=chinese_font)
        self.axes[0, 1].set_title('平均最大数字', fontproperties=chinese_font)
        self.axes[1, 0].set_title('平均损失', fontproperties=chinese_font)
        self.axes[1, 1].set_title('平均奖励', fontproperties=chinese_font)
        self.axes[2, 0].set_title('平均梯度范数', fontproperties=chinese_font)

        # 设置坐标轴标签 - 使用中文字体
        for ax in self.axes.flat:
            ax.set_xlabel('训练轮次', fontproperties=chinese_font)
        
        self.axes[0, 0].set_ylabel('分数', fontproperties=chinese_font)
        self.axes[0, 1].set_ylabel('最大数字', fontproperties=chinese_font)
        self.axes[1, 0].set_ylabel('损失值', fontproperties=chinese_font)
        self.axes[1, 1].set_ylabel('奖励值', fontproperties=chinese_font)
        self.axes[2, 0].set_ylabel('梯度范数', fontproperties=chinese_font)

        # 绘制数据
        self.axes[0, 0].plot(self.episodes, self.scores, 'b-')
        self.axes[0, 1].plot(self.episodes, self.max_tiles, 'g-')
        
        if self.losses:
            self.axes[1, 0].plot(self.episodes[:len(self.losses)], self.losses, 'r-')
        
        if self.grads:
            self.axes[2, 0].plot(self.episodes[:len(self.grads)], self.grads, 'p-')

        if self.rewards:
            self.axes[1, 1].plot(self.episodes[:len(self.rewards)], self.rewards, 'y-')
        
        # 保存图表
        plt.savefig(filename)
        plt.close(self.fig)  # 关闭图表释放资源

class GameVisualizer:
    """游戏过程可视化类"""
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.root.title('2048 AI 可视化')
        else:
            self.root = root
        
        # 创建游戏状态显示框架
        self.game_frame = Frame(self.root, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        self.game_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建信息显示框架
        self.info_frame = Frame(self.root, width=300, height=c.SIZE)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 创建网格单元格
        self.grid_cells = []
        self.init_grid()
        
        # 创建信息标签 - 使用支持中文的字体
        self.score_label = Label(self.info_frame, text="分数: 0", font=("Microsoft YaHei", 12))
        self.score_label.pack(pady=5)
        
        self.max_tile_label = Label(self.info_frame, text="最大数字: 0", font=("Microsoft YaHei", 12))
        self.max_tile_label.pack(pady=5)
        
        self.step_label = Label(self.info_frame, text="步数: 0", font=("Microsoft YaHei", 12))
        self.step_label.pack(pady=5)
        
        self.action_label = Label(self.info_frame, text="动作: 无", font=("Microsoft YaHei", 12))
        self.action_label.pack(pady=5)
        
        # 创建Q值可视化
        self.fig = Figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.info_frame)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # 设置图表标题和标签 - 使用中文字体
        self.ax.set_title('动作Q值', fontproperties=chinese_font)
        self.ax.set_xlabel('动作', fontproperties=chinese_font)
        self.ax.set_ylabel('Q值', fontproperties=chinese_font)
        self.ax.set_xticks([0, 1, 2, 3])
        self.ax.set_xticklabels(['上', '下', '左', '右'], fontproperties=chinese_font)
        
        self.q_bars = self.ax.bar([0, 1, 2, 3], [0, 0, 0, 0], color='skyblue')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def init_grid(self):
        """初始化游戏网格"""
        background = Frame(self.game_frame, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()
        
        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)
    
    def update_grid_cells(self, matrix):
        """更新网格单元格显示"""
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT.get(new_number, c.BACKGROUND_COLOR_CELL_EMPTY),
                        fg=c.CELL_COLOR_DICT.get(new_number, "black")
                    )
        self.game_frame.update_idletasks()
    
    def update_info(self, score, max_tile, step, action=None):
        """更新信息显示"""
        self.score_label.config(text=f"分数: {score}")
        self.max_tile_label.config(text=f"最大数字: {max_tile}")
        self.step_label.config(text=f"步数: {step}")
        
        if action is not None:
            action_text = ['上', '下', '左', '右'][action]
            self.action_label.config(text=f"动作: {action_text}")
    
    def update_q_values(self, q_values):
        """更新Q值可视化"""
        for i, bar in enumerate(self.q_bars):
            bar.set_height(q_values[i])
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
    
    def mainloop(self):
        """启动主循环"""
        self.root.mainloop()
    
    def update(self):
        """更新界面"""
        self.root.update()