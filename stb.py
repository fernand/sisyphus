import argparse
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.sac import SAC

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true')
parser.add_argument('--episodes', type=int, default=5000)
args = parser.parse_args()

ENV_ID = 'BipedalWalker-v3'
MAX_STEPS  = 1600

render_mode = 'human' if args.render else None
env = gym.make(ENV_ID, render_mode=render_mode)
env = gym.wrappers.RecordEpisodeStatistics(env) # keeps episode returns

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=512,
    train_freq=(1, "step"),
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
)

class TqdmCallback(BaseCallback):
    def _on_training_start(self):
        self.ep = 0
        self.recent = []

    def _on_step(self):
        if self.locals["dones"][0]:
            self.ep += 1
            ep_ret = self.locals["infos"][0]["episode"]["r"]
            self.recent.append(ep_ret)
            if len(self.recent) > 10:
                self.recent.pop(0)

            if self.ep % 10 == 0:
                buf_size = self.model.replay_buffer.size()
                avg_ret = sum(self.recent) / len(self.recent)
                print(f"Episode {self.ep:4d} | Return {ep_ret:7.1f} "
                      f"| Avg {avg_ret:7.1f} | Buffer {buf_size:,}")

            if self.ep >= args.episodes:
                return False
        return True

total_steps = args.episodes * MAX_STEPS
model.learn(total_timesteps=total_steps, callback=TqdmCallback())
env.close()