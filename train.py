import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

parser = argparse.ArgumentParser()
parser.add_argument('--render',        action='store_true', help='turn on live viewer')
parser.add_argument('--render_every',  type=int, default=1, help='draw 1 in N steps')
args = parser.parse_args()

device, ENV_ID = torch.device('cpu'), 'BipedalWalker-v3'
GAMMA, LAMBDA, ACTOR_LR, CRITIC_LR = 0.99, 0.95, 3e-4, 3e-4
MAX_EPISODES, MAX_STEPS, HIDDEN_SIZES = 2000, 1600, (128, 64)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h1, h2 = HIDDEN_SIZES
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh())
        self.mu_head     = nn.Linear(h2, act_dim)
        self.logstd_head = nn.Parameter(torch.zeros(act_dim))
        self.v_head      = nn.Linear(h2, 1)

        # Separate parameters for actor and critic
        self.actor_params = list(self.shared.parameters()) + list(self.mu_head.parameters()) + [self.logstd_head]
        self.critic_params = list(self.shared.parameters()) + list(self.v_head.parameters())
    def forward(self, x):
        feat = self.shared(x)
        mu, v = self.mu_head(feat), self.v_head(feat).squeeze(-1)
        return mu, self.logstd_head.exp(), v
    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mu, std, v = self(obs_t)
        act = Normal(mu, std).sample()
        return act.numpy(), v.item()

# eligibility-trace helper
def update(trace, new_grad, gamma, lam):
    return gamma * lam * trace + new_grad

def train():
    render_mode = 'human' if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim)
    # Separate optimizers for actor and critic parameters
    actor_opt = torch.optim.Adam(net.actor_params, ACTOR_LR)
    critic_opt = torch.optim.Adam(net.critic_params, CRITIC_LR)
    actor_tr = [torch.zeros_like(p) for p in net.actor_params]
    critic_tr = [torch.zeros_like(p) for p in net.critic_params]

    for ep in trange(MAX_EPISODES, desc='episodes'):
        obs, _ = env.reset(seed=None)
        ep_ret = 0.0
        for t in actor_tr: t.zero_()
        for t in critic_tr: t.zero_()

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0):
                env.render()

            act, v = net.act(obs)
            act = np.clip(act, env.action_space.low, env.action_space.high)
            obs_next, r, term, trunc, _ = env.step(act)
            done = term or trunc
            # Compute target value
            with torch.no_grad():
                obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32)
                _, _, v_next = net(obs_next_t)
                target = r + GAMMA * (0.0 if done else v_next)
                δ = target - v  # TD-error for actor

            # critic update
            critic_opt.zero_grad()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            _, _, v_pred = net(obs_t)
            critic_loss = 0.5 * (v_pred - target)**2
            critic_loss.backward()

            # Update critic eligibility traces
            with torch.no_grad():
                for p, z in zip(net.critic_params, critic_tr):
                    if p.grad is not None:
                        orig_grad = p.grad.clone()
                        z[:] = update(z, orig_grad, GAMMA, LAMBDA)
                        p.grad = z.clone()
            torch.nn.utils.clip_grad_norm_(net.critic_params, 0.5)
            critic_opt.step()

            # actor update
            actor_opt.zero_grad()
            mu, std, _ = net(obs_t)  # reuse obs_t from critic
            dist = Normal(mu, std)
            act_t = torch.from_numpy(act)
            logp = dist.log_prob(act_t).sum()
            logp.backward()

            # Update actor eligibility traces and scale by TD error
            with torch.no_grad():
                for p, z in zip(net.actor_params, actor_tr):
                    if p.grad is not None:
                        orig_grad = p.grad.clone()
                        z[:] = update(z, orig_grad, GAMMA, LAMBDA)
                        p.grad = δ.item() * z  # Use δ.item() to get scalar value
            torch.nn.utils.clip_grad_norm_(net.actor_params, 0.5)
            actor_opt.step()

            ep_ret += r
            if done: break
            obs = obs_next

        if ep % 10 == 0:
            print(f'Episode {ep:4d} | Return {ep_ret:7.1f}')

    env.close()

if __name__ == '__main__':
    train()
