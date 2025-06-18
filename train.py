import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from tqdm import trange

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='turn on live viewer')
parser.add_argument('--render_every', type=int, default=1, help='draw 1 in N steps')
args = parser.parse_args()

ENV_ID = 'BipedalWalker-v3'
GAMMA = 0.99
ENTROPY_COEF = 0.001  # Reduced for more exploitation
ACTOR_LR = 3e-4  # Increased for faster learning
CRITIC_LR = 3e-4
MAX_EPISODES = 5000
MAX_STEPS = 1600
HIDDEN_SIZES = (256, 256)
DEVICE = torch.device('cpu')

def make_dist(mu, std):
    base = Normal(mu, std)
    return TransformedDistribution(base, [TanhTransform(cache_size=1)])

class ActorCritic(nn.Module):
    """Actor-Critic with separate networks for stability."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        h1, h2 = HIDDEN_SIZES

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(h2, act_dim)
        self.logstd_head = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, 1)
        )

        self.actor_params = list(self.actor.parameters()) + list(self.mu_head.parameters()) + [self.logstd_head]
        self.critic_params = list(self.critic.parameters())

    def forward(self, x: torch.Tensor):
        # Actor
        actor_feat = self.actor(x)
        mu = self.mu_head(actor_feat)
        std = self.logstd_head.exp()

        # Critic
        v = self.critic(x).squeeze(-1)

        return mu, std, v

    @torch.no_grad()
    def act(self, obs):
        mu, std, v = self(obs)
        act = make_dist(mu, std).rsample()
        return act, v.item()

def init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            gain = np.sqrt(2)
            nn.init.orthogonal_(m.weight, gain)
            nn.init.constant_(m.bias, 0.)
    # Small output layers
    nn.init.orthogonal_(net.mu_head.weight, 0.01)
    nn.init.constant_(net.mu_head.bias, 0.)
    nn.init.orthogonal_(net.critic[-1].weight, 1.0)
    nn.init.constant_(net.critic[-1].bias, 0.)

def train():
    render_mode = 'human' if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = ActorCritic(obs_dim, act_dim).to(DEVICE)
    init(net)

    actor_opt = torch.optim.Adam(net.actor_params, ACTOR_LR)
    critic_opt = torch.optim.Adam(net.critic_params, CRITIC_LR)

    recent_returns = []
    low, high = env.observation_space.low, env.observation_space.high
    obs_mean = torch.from_numpy((low + high) / 2).to(DEVICE)
    obs_scale = torch.from_numpy((high - low) / 2).to(DEVICE)

    for ep in trange(MAX_EPISODES, desc='episodes'):
        obs, _ = env.reset()
        ep_ret = 0.0
        steps_survived = 0

        if ep % 50 == 0 and ep > 0:
            for param_group in actor_opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.98, 1e-5)
            for param_group in critic_opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.98, 1e-5)

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0):
                env.render()

            # Normalize observations to [-1, 1]
            obs_t = (torch.as_tensor(obs, dtype=torch.float32, device=DEVICE) - obs_mean) / obs_scale

            # Action selection
            act, _ = net.act(obs_t)
            obs_next, r, term, trunc, _ = env.step(act.cpu().numpy())
            done = term or trunc

            obs_next_t = (torch.as_tensor(obs_next, dtype=torch.float32, device=DEVICE) - obs_mean) / obs_scale

            mu, std, v = net(obs_t)
            with torch.no_grad():
                _, _, v_next = net(obs_next_t)
            v_next = v_next.detach()
            delta = (r + GAMMA * (1.0 - float(done)) * v_next) - v

            critic_opt.zero_grad()
            critic_loss = 0.5 * delta.pow(2).mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.critic_params, 0.5)
            critic_opt.step()

            actor_opt.zero_grad()
            dist = make_dist(mu, std)
            act_t = torch.as_tensor(act, dtype=torch.float32, device=DEVICE)
            log_prob = dist.log_prob(torch.clamp(act_t, -0.999, 0.999)).sum()
            entropy = dist.base_dist.entropy().sum()
            actor_loss = -log_prob * delta.detach() - ENTROPY_COEF * entropy
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.actor_params, 0.5)
            actor_opt.step()

            ep_ret += r
            steps_survived += 1

            if done:
                break
            obs = obs_next

        recent_returns.append(ep_ret)
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        if ep % 10 == 0:
            avg_ret = np.mean(recent_returns[-10:]) if len(recent_returns) >= 10 else ep_ret
            print(f'Episode {ep:4d} | Return {ep_ret:7.1f} | Avg {avg_ret:7.1f} | Steps {steps_survived:4d} | σ {net.logstd_head.exp().mean():.3f} | LR {round(critic_opt.param_groups[0]['lr'], 5)}')

    env.close()

if __name__ == '__main__':
    train()