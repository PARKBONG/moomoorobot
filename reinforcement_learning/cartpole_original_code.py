# conda activate cobot_py311
# pip install gymnasium torch matplotlib

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import glob
import torch.nn.functional as F

# 하이퍼 파라미터
gamma = 0.99
learning_rate = 0.0005
batch_size = 100
memory_size = 5000
episodes = 5000
# ϵ-greedy 사용 시, 필요
# epsilon_start = 1.0
# epsilon_end = 0.001
# epsilon_decay = 0.995

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state, target_net, action_dim):
	# ϵ-greedy
	# if random.random() < epsilon:
    #     return random.randint(0, action_dim - 1)
    # else:
    #     return target_net(state).argmax().item()
    
    q_value = target_net(state)
    p = F.softmax(q_value, dim=0).tolist()
    p = np.array(p)
    p /= p.sum()
    action = np.random.choice(action_dim, p=p)
    return action

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    
    state_batch = torch.stack(batch[0])
    action_batch = torch.tensor(batch[1]).unsqueeze(1)
    reward_batch = torch.tensor(batch[2])
    next_state_batch = torch.stack(batch[3])
    done_batch = torch.tensor(batch[4], dtype=torch.float32)
    
    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    # DQN
    target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def draw_cartpole(ax, state, title="CartPole State"):
    """matplotlib으로 CartPole 상태를 그리는 함수"""
    # state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    cart_x = state[0]
    pole_angle = state[2]
    
    # 카트 그리기
    cart_width = 0.3
    cart_height = 0.15
    cart = plt.Rectangle((cart_x - cart_width/2, -cart_height/2), 
                         cart_width, cart_height, 
                         color='blue', fill=True)
    ax.add_patch(cart)
    
    # 폴(막대) 그리기
    pole_length = 0.5
    pole_x_end = cart_x + pole_length * np.sin(pole_angle)
    pole_y_end = pole_length * np.cos(pole_angle)
    
    ax.plot([cart_x, pole_x_end], [0, pole_y_end], 'r-', linewidth=3)
    ax.plot([pole_x_end], [pole_y_end], 'ro', markersize=8)
    
    # 축 설정
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Cart Position')
    ax.set_ylabel('Height')
    ax.set_title(title)
    
    # 상태 정보 표시
    info_text = f"Position: {cart_x:.2f}\nAngle: {np.degrees(pole_angle):.1f}°"
    ax.text(-2.3, 0.8, info_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_size)

# epsilon = epsilon_start

episode_rewards = []
episode_reward = 0

current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "dqn_saved_models")
os.makedirs(save_dir, exist_ok=True)

# matplotlib 실시간 렌더링 설정
render_interval = 100  # 100 에피소드마다 그래프 업데이트
cartpole_render_interval = 50  # 50 에피소드마다 CartPole 시각화

plt.ion()  # Interactive mode 활성화
fig = plt.figure(figsize=(16, 5))
ax1 = plt.subplot(131)  # 리워드 그래프
ax2 = plt.subplot(132)  # 이동평균
ax3 = plt.subplot(133)  # CartPole 시각화

for episode in range(episodes):
    state = torch.tensor(env.reset()[0], dtype=torch.float32)
    state_np = state.numpy()
    
    if episode % 100 == 0: 
        print(f"Episode {episode}, Avg Reward: {episode_reward/100}")
    if episode % 100 == 0 :
        episode_reward = 0
    total_reward = 0
    last_state = state_np

    # 500 초과인 경우는 done으로 판단
    while total_reward < 501 :
        action = select_action(state, target_net, action_dim)

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        memory.push((state, action, reward, next_state, done))

        state = next_state
        last_state = state.numpy()  # 마지막 상태 저장
        total_reward += reward

        optimize_model(memory, policy_net, target_net, optimizer)
        
        if done :
            break
    
    # 500점 달성한 모델 저장
    if total_reward >= 500 :
        model_path = os.path.join(save_dir, f"dqn_model_episode_{episode}.pth")
        torch.save(policy_net.state_dict(), model_path)
    
    episode_reward += total_reward
    
    # ϵ-greedy 사용 시, 필요
  	# if episode % 10 == 0 :
    #     epsilon = max(epsilon_end, epsilon*epsilon_decay)
    
    if episode % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    
    # matplotlib으로 실시간 시각화
    if (episode % render_interval == 0) and (episode > 0):
        ax1.clear()
        ax1.plot(episode_rewards, linewidth=2, color='blue')
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Total Reward', fontsize=10)
        ax1.set_title(f'Training Progress (Episode {episode})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 평균 리워드 표시
        ax2.clear()
        window = 50
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(moving_avg, linewidth=2, color='red', label=f'{window}-episode Avg')
            ax2.set_xlabel('Episode', fontsize=10)
            ax2.set_ylabel('Avg Reward', fontsize=10)
            ax2.set_title(f'Moving Average', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # CartPole 시각화
        ax3.clear()
        draw_cartpole(ax3, last_state, f'Latest CartPole (Episode {episode})')
        
        plt.tight_layout()
        plt.pause(0.01)  # 0.01초 일시 중지 (non-blocking)

plt.ioff()  # Interactive mode 종료
plt.show()  # 최종 그래프 표시

print("\n=== 학습 완료! ===")
print(f"최종 평균 리워드: {np.mean(episode_rewards[-100:]):.2f}")
print(f"저장된 모델 수: {len(glob.glob(os.path.join(save_dir, '*.pth')))}")

# 테스트 진행 (선택사항 - 필요 시에만 실행)
# print("\n=== 테스트 시작 ===")
# test_env = gym.make("CartPole-v1")
# model_paths = glob.glob(os.path.join(save_dir, "*.pth"))

# for model_path in model_paths:
#     policy_net.load_state_dict(torch.load(model_path))
#     policy_net.eval()
#     
#     total_reward = 0
#     state = torch.tensor(test_env.reset()[0], dtype=torch.float32)
#     
#     while total_reward < 501:
#         with torch.no_grad():
#             action = policy_net(state).argmax().item()
#         
#         next_state, reward, done, _, _ = test_env.step(action)
#         state = torch.tensor(next_state, dtype=torch.float32)
#         total_reward += reward
#         
#         if done:
#             break
#     
#     print(f"Model test - Reward: {total_reward}")
# 
# test_env.close()