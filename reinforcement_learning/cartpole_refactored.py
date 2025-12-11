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
import threading
import time
from pathlib import Path
from typing import Tuple, Optional

# ============================================================================
# 1. DQN Network
# ============================================================================
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# 2. Replay Memory
# ============================================================================
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# ============================================================================
# 3. Environment Wrapper
# ============================================================================
class CartPoleEnv:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        return torch.tensor(state, dtype=torch.float32), state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, np.ndarray]:
        next_state, reward, done, _, _ = self.env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        return next_state_tensor, reward, done, next_state

    def close(self):
        self.env.close()


# ============================================================================
# 4. DQN Trainer (Headless - 학습만 담당)
# ============================================================================
class DQNTrainer:
    def __init__(self, env: CartPoleEnv, model_save_dir: str, 
                 gamma: float = 0.99, lr: float = 0.0005,
                 batch_size: int = 100, memory_size: int = 5000):
        self.env = env
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        # 네트워크
        self.policy_net = DQN(env.state_dim, env.action_dim)
        self.target_net = DQN(env.state_dim, env.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 최적화기
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 메모리
        self.memory = ReplayMemory(memory_size)

        # 통계
        self.episode_rewards = []
        self.step_count = 0

    def select_action(self, state: torch.Tensor) -> int:
        """Softmax 기반 액션 선택"""
        with torch.no_grad():
            q_value = self.target_net(state)
            p = F.softmax(q_value, dim=0).numpy()
            p /= p.sum()
            action = np.random.choice(self.env.action_dim, p=p)
        return int(action)

    def optimize_model(self):
        """DQN 손실 함수로 모델 업데이트"""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.stack(batch[0])
        action_batch = torch.tensor(batch[1]).unsqueeze(1)
        reward_batch = torch.tensor(batch[2])
        next_state_batch = torch.stack(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self) -> float:
        """한 에피소드 학습"""
        state, _ = self.env.reset()
        total_reward = 0

        while total_reward < 501:
            action = self.select_action(state)
            next_state, reward, done, next_state_np = self.env.step(action)

            self.memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            self.step_count += 1

            self.optimize_model()

            if done:
                break

        # 주기적으로 target_net 업데이트
        if len(self.episode_rewards) % 20 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_reward

    def save_model(self, episode: int):
        """모델 저장"""
        model_path = self.model_save_dir / f"dqn_episode_{episode:05d}_reward_{self.episode_rewards[-1]:.0f}.pth"
        torch.save(self.policy_net.state_dict(), model_path)

    def train(self, num_episodes: int, save_interval: int = 100):
        """메인 학습 루프"""
        print(f"🚀 Training started - {num_episodes} episodes")
        print(f"📁 Model save directory: {self.model_save_dir}")

        for episode in range(num_episodes):
            reward = self.train_episode()
            self.episode_rewards.append(reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"[Episode {episode + 1:4d}] Avg Reward (100): {avg_reward:6.2f} | Total steps: {self.step_count}")

            if (episode + 1) % save_interval == 0:
                self.save_model(episode + 1)

        print(f"\n✅ Training completed!")
        print(f"Final avg reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Total steps: {self.step_count}")


# ============================================================================
# 5. CartPole Visualizer (matplotlib)
# ============================================================================
def draw_cartpole(ax, state: np.ndarray, title: str = "CartPole State"):
    """matplotlib으로 CartPole 상태 그리기"""
    cart_x = state[0]
    pole_angle = state[2]

    # 카트 그리기
    cart_width = 0.3
    cart_height = 0.15
    cart = plt.Rectangle((cart_x - cart_width / 2, -cart_height / 2),
                         cart_width, cart_height, color='blue', fill=True)
    ax.add_patch(cart)

    # 폴(막대) 그리기
    pole_length = 0.5
    pole_x_end = cart_x + pole_length * np.sin(pole_angle)
    pole_y_end = pole_length * np.cos(pole_angle)

    ax.plot([cart_x, pole_x_end], [0, pole_y_end], 'r-', linewidth=3)
    ax.plot([pole_x_end], [pole_y_end], 'ro', markersize=8)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Height')
    ax.set_title(title, fontsize=11)

    info_text = f"Pos: {cart_x:.2f}\nAngle: {np.degrees(pole_angle):.1f}°"
    ax.text(-2.3, 0.8, info_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ============================================================================
# 6. Inference + Visualizer (백그라운드 스레드)
# ============================================================================
class DQNVisualizer(threading.Thread):
    def __init__(self, model_dir: str, state_dim: int, action_dim: int):
        super().__init__(daemon=True)
        self.model_dir = Path(model_dir)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.running = True

        # 네트워크
        self.net = DQN(state_dim, action_dim)
        self.net.eval()

        # 환경 (headless)
        self.env = gym.make("CartPole-v1")

        # 통계
        self.episode_count = 0
        self.latest_reward = 0
        self.latest_state = None
        self.latest_state_history = []  # 상태 히스토리 저장
        self.replay_progress = 0  # 리플레이 진행도 (0.0 ~ 1.0)

        # 시각화 데이터 (메인 스레드에서 접근)
        self.episode_rewards = []
        self.last_model_path = None
        self.new_data_available = False

    def load_latest_model(self) -> bool:
        """최신 모델 파일 로드"""
        model_files = sorted(self.model_dir.glob("*.pth"))
        if not model_files:
            return False

        latest_model = model_files[-1]

        # 이미 로드한 모델이면 스킵
        if latest_model == self.last_model_path:
            return False

        try:
            self.net.load_state_dict(torch.load(latest_model, weights_only=True))
            self.last_model_path = latest_model
            
            # 파일명에서 에피소드 수와 리워드 추출
            filename = latest_model.stem
            if "reward" in filename:
                reward_str = filename.split("reward_")[-1]
                self.latest_reward = float(reward_str)
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def inference_episode(self) -> Tuple[float, list]:
        """추론 에피소드 실행 (상태 히스토리 저장)"""
        state, _ = self.env.reset()
        total_reward = 0
        state_history = [state.copy()]  # 초기 상태 저장

        while total_reward < 501:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.net(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, _, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            state_history.append(state.copy())  # 모든 상태 저장
            self.latest_state = state

            if done:
                break

        return total_reward, state_history

    def run(self):
        """백그라운드 추론 루프 (matplotlib 제외)"""
        print("\n🎨 Inference thread started - waiting for models...")

        while self.running:
            # 최신 모델 로드 시도
            if self.load_latest_model():
                # 추론 실행 (상태 히스토리 포함)
                reward, state_history = self.inference_episode()
                self.episode_rewards.append(reward)
                self.latest_state_history = state_history
                self.replay_progress = 0
                self.episode_count += 1

                print(f"[Inference {self.episode_count}] Reward: {reward:.0f} (Steps: {len(state_history)})")
                self.new_data_available = True

            time.sleep(1)  # 1초마다 체크

    def stop(self):
        """스레드 종료"""
        self.running = False
        self.env.close()


# ============================================================================
# 7. Visualization Updater (별도 스레드)
# ============================================================================
class VisualizationUpdater(threading.Thread):
    def __init__(self, ax1, ax2, ax3, trainer: DQNTrainer, visualizer: DQNVisualizer):
        super().__init__(daemon=True)
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.trainer = trainer
        self.visualizer = visualizer
        self.running = True
        
        self.replay_frame_idx = 0
        self.last_episode_count = 0

    def run(self):
        """시각화 업데이트 루프 (별도 스레드)"""
        print("🎨 Visualization updater thread started...")
        
        while self.running:
            try:
                # CartPole 프레임 진행
                if self.visualizer.latest_state_history:
                    if self.visualizer.episode_count != self.last_episode_count:
                        self.replay_frame_idx = 0
                        self.last_episode_count = self.visualizer.episode_count
                        print(f"✨ New replay loaded! (Episode {self.visualizer.episode_count})")
                    
                    if self.replay_frame_idx >= len(self.visualizer.latest_state_history):
                        self.replay_frame_idx = 0
                    
                    self.replay_frame_idx += 1
                
                # matplotlib 업데이트 (스레드에서)
                self.ax1.clear()
                if self.trainer.episode_rewards:
                    self.ax1.plot(self.trainer.episode_rewards, linewidth=2, color='blue', alpha=0.7)
                    self.ax1.set_xlabel('Training Episode')
                    self.ax1.set_ylabel('Total Reward')
                    self.ax1.set_title(f'Training Progress (Episode {len(self.trainer.episode_rewards)})')
                    self.ax1.grid(True, alpha=0.3)

                self.ax2.clear()
                if len(self.trainer.episode_rewards) >= 50:
                    window = 50
                    moving_avg = np.convolve(self.trainer.episode_rewards, np.ones(window) / window, mode='valid')
                    self.ax2.plot(moving_avg, linewidth=2, color='red')
                    self.ax2.set_xlabel('Training Episode')
                    self.ax2.set_ylabel(f'{window}-Episode Avg Reward')
                    self.ax2.set_title(f'Training Moving Average: {moving_avg[-1]:.2f}')
                    self.ax2.grid(True, alpha=0.3)

                self.ax3.clear()
                if self.visualizer.latest_state_history:
                    current_idx = min(self.replay_frame_idx - 1, len(self.visualizer.latest_state_history) - 1)
                    current_state = self.visualizer.latest_state_history[current_idx]
                    progress_percent = (current_idx / len(self.visualizer.latest_state_history)) * 100
                    
                    draw_cartpole(self.ax3, current_state,
                                f'CartPole Replay (#{self.visualizer.episode_count})\n'
                                f'Frame: {current_idx}/{len(self.visualizer.latest_state_history)-1} ({progress_percent:.0f}%)\n'
                                f'Reward: {self.visualizer.latest_reward:.0f}')
                else:
                    self.ax3.text(0.5, 0.5, 'Waiting for inference...',
                                ha='center', va='center', fontsize=14)
                    self.ax3.set_xlim(-3, 3)
                    self.ax3.set_ylim(-1, 1)

                plt.tight_layout()
                plt.pause(0.05)  # 50ms 업데이트

            except Exception as e:
                print(f"⚠️ Visualization update error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False



# ============================================================================
# 8. Main
# ============================================================================
def main():
    # 설정
    NUM_EPISODES = 5000
    SAVE_INTERVAL = 100
    MODEL_DIR = "dqn_saved_models"

    # 환경 생성
    env = CartPoleEnv()
    print(f"📊 Environment: CartPole-v1")
    print(f"   State Dim: {env.state_dim}")
    print(f"   Action Dim: {env.action_dim}")

    # 학습기 생성
    trainer = DQNTrainer(env, MODEL_DIR)

    # 시각화 스레드 시작 (추론 담당)
    visualizer = DQNVisualizer(MODEL_DIR, env.state_dim, env.action_dim)
    visualizer.start()

    try:
        print(f"\n🚀 Starting training and visualization...")
        print(f"💡 Training runs in main thread")
        print(f"🎨 Visualization updates in separate thread")
        
        # matplotlib 윈도우 생성 (메인 스레드에서)
        plt.ion()
        fig = plt.figure(figsize=(16, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        # 시각화 업데이터 스레드 시작
        vis_updater = VisualizationUpdater(ax1, ax2, ax3, trainer, visualizer)
        vis_updater.start()
        
        # 학습 루프 (메인 스레드)
        print(f"[Training] Starting {NUM_EPISODES} episodes...")
        for episode in range(NUM_EPISODES):
            reward = trainer.train_episode()
            trainer.episode_rewards.append(reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(trainer.episode_rewards[-100:])
                print(f"[Episode {episode + 1:4d}] Avg Reward (100): {avg_reward:6.2f} | Total steps: {trainer.step_count}")

            if (episode + 1) % SAVE_INTERVAL == 0:
                trainer.save_model(episode + 1)

        print(f"\n✅ Training completed!")
        print(f"Final avg reward: {np.mean(trainer.episode_rewards[-100:]):.2f}")
        
        # 시각화 계속 표시
        print("📊 Visualization will continue updating. Close the window to exit.")
        vis_updater.running = True  # 계속 실행
        plt.show()  # 블로킹

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        print("🛑 Stopping all threads...")
        vis_updater.stop()
        visualizer.stop()
        visualizer.join(timeout=2)
        env.close()
        plt.close('all')
        print("✅ All completed!")


if __name__ == "__main__":
    main()
