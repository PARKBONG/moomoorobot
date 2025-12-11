import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
import json

# ============================================================================
# DQN Network
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

    def save(self, path: Path, episode: int, reward: float):
        """모델 저장"""
        path.mkdir(exist_ok=True)
        model_path = path / f"dqn_episode_{episode:05d}_reward_{reward:.0f}.pth"
        torch.save(self.state_dict(), model_path)
        return model_path

    def load(self, path: Path) -> bool:
        """모델 로드"""
        try:
            self.load_state_dict(torch.load(path, weights_only=True))
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    @staticmethod
    def load_latest(model_dir: Path) -> Optional[Tuple[Path, 'DQN']]:
        """최신 모델 파일 찾아서 로드"""
        model_files = sorted(model_dir.glob("dqn_episode_*.pth"))
        if not model_files:
            return None
        
        latest_model = model_files[-1]
        # 에피소드 번호 추출
        try:
            episode_str = latest_model.stem.split("_")[2]
            episode = int(episode_str)
        except:
            episode = 0
        
        return latest_model, episode

# ============================================================================
# Replay Memory
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
# DQN Trainer
# ============================================================================
class DQNTrainer:
    def __init__(self, env, model_save_dir: str, 
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

    def train_episode(self) -> Tuple[float, dict]:
        """한 에피소드 학습 (trajectory 데이터 포함)"""
        state, state_np = self.env.reset()
        total_reward = 0
        
        # Trajectory 추적
        states = [state_np.copy()]
        actions = []
        rewards = []
        
        while total_reward < 501:
            action = self.select_action(state)
            next_state, reward, done, next_state_np = self.env.step(action)

            self.memory.push((state, action, reward, next_state, done))
            
            # Trajectory 기록
            actions.append(int(action))
            rewards.append(float(reward))
            
            state = next_state
            total_reward += reward
            self.step_count += 1

            self.optimize_model()

            if done:
                break
            
            states.append(next_state_np.copy())

        # 주기적으로 target_net 업데이트
        if len(self.episode_rewards) % 20 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Trajectory 데이터
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'timesteps': len(actions)
        }
        
        return total_reward, trajectory

    def save_model(self, episode: int):
        """모델 저장 (에피소드 번호와 리워드 포함)"""
        reward = self.episode_rewards[-1]
        model_path = self.policy_net.save(self.model_save_dir, episode, reward)
        print(f"💾 Model saved: {model_path.name}")

    def save_rewards(self):
        """모든 에피소드 리워드를 JSON으로 저장"""
        reward_path = self.model_save_dir / "episode_rewards.json"
        with open(reward_path, 'w') as f:
            json.dump(self.episode_rewards, f)

    def save_trajectory(self, episode: int, trajectory: dict):
        """특정 에피소드의 trajectory 저장 (state/action/reward)"""
        trajectory_path = self.model_save_dir / f"episode_{episode:05d}_trajectory.json"
        
        # states를 리스트로 변환
        states_data = trajectory['states']
        if isinstance(states_data, np.ndarray):
            states_data = states_data.tolist()
        elif isinstance(states_data, list) and len(states_data) > 0 and isinstance(states_data[0], np.ndarray):
            states_data = [s.tolist() for s in states_data]
        
        trajectory_data = {
            'episode': episode,
            'total_reward': trajectory['total_reward'],
            'timesteps': trajectory['timesteps'],
            'states': states_data,
            'actions': trajectory['actions'],
            'rewards': trajectory['rewards']
        }
        
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory_data, f)
        
        print(f"📹 Trajectory saved: {trajectory_path.name}")

    def train(self, num_episodes: int, save_interval: int = 100):
        """메인 학습 루프"""
        print(f"\n{'='*70}")
        print(f"🚀 DQN Training Started")
        print(f"{'='*70}")
        print(f"📊 Environment: CartPole-v1")
        print(f"📁 Model save directory: {self.model_save_dir}")
        print(f"🎯 Total episodes: {num_episodes}")
        print(f"💾 Save interval: {save_interval}")
        print(f"{'='*70}\n")

        for episode in range(num_episodes):
            reward, trajectory = self.train_episode()
            self.episode_rewards.append(reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"[Episode {episode + 1:5d}] Avg Reward (100): {avg_reward:6.2f} | "
                      f"Total steps: {self.step_count:7d}")

            if (episode + 1) % save_interval == 0:
                self.save_model(episode + 1)
                self.save_trajectory(episode + 1, trajectory)  # 🆕 trajectory 저장
                self.save_rewards()  # 리워드도 함께 저장

        print(f"\n{'='*70}")
        print(f"✅ Training Completed!")
        print(f"{'='*70}")
        print(f"Final avg reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Total steps: {self.step_count}")
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"{'='*70}\n")
        
        self.save_rewards()  # 최종 리워드 저장