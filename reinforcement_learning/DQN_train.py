# conda activate cobot_py311
# pip install gymnasium torch matplotlib

from agent.dqn import DQNTrainer
from env.cartpole import CartPole


NUM_EPISODES = 5000
SAVE_INTERVAL = 100
MODEL_DIR = "dqn_saved_models"

# 환경 생성
env = CartPole()

# 학습기 생성
trainer = DQNTrainer(env, MODEL_DIR)

try:
    # 학습 실행
    trainer.train(NUM_EPISODES, SAVE_INTERVAL)

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    print(f"Episodes completed: {len(trainer.episode_rewards)}")
    trainer.save_rewards()
finally:
    env.close()
    print("✅ All completed!")

