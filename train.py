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
parser.add_argument('--render', action='store_true', help='turn on live viewer')
parser.add_argument('--render_every', type=int, default=1, help='draw 1 in N steps')
args = parser.parse_args()

ENV_ID      = 'BipedalWalker-v3'
GAMMA       = 0.99
LAMBDA      = 0.95
ACTOR_LR    = 1e-4
CRITIC_LR   = 1e-4
MAX_EPISODES = 2000
MAX_STEPS    = 1600
HIDDEN_SIZES = (128, 64)
DEVICE       = torch.device('cpu')

class ActorCritic(nn.Module):
    """Shared torso, Gaussian actor, scalar critic."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        h1, h2 = HIDDEN_SIZES
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
        )
        self.mu_head     = nn.Linear(h2, act_dim)
        self.logstd_head = nn.Parameter(torch.zeros(act_dim))
        self.v_head      = nn.Linear(h2, 1)
        self.shared_params: list[nn.Parameter] = list(self.shared.parameters())
        self.actor_params:  list[nn.Parameter] = list(self.mu_head.parameters()) + [self.logstd_head]
        self.critic_params: list[nn.Parameter] = self.shared_params + list(self.v_head.parameters())

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        mu   = self.mu_head(feat)
        std  = self.logstd_head.exp()
        v    = self.v_head(feat).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        mu, std, v = self(obs_t)
        act = Normal(mu, std).sample()
        return act.cpu().numpy(), v.item()

def decay_and_add(trace: torch.Tensor, grad: torch.Tensor):
    """Eligibility update: z ← γλ z + g_t"""
    trace.mul_(GAMMA * LAMBDA).add_(grad)

def apply_traces(params, traces, scale: float = 1.0):
    """Scales grads by eligibility traces before the optimiser step."""
    with torch.no_grad():
        for p, z in zip(params, traces):
            if p.grad is None:
                continue
            decay_and_add(z, p.grad)
            p.grad.copy_(z).mul_(scale)

def train():
    render_mode = 'human' if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = ActorCritic(obs_dim, act_dim).to(DEVICE)

    # Two optimisers, shared trunk only in critic optimiser
    actor_opt  = torch.optim.Adam(net.actor_params,  ACTOR_LR)
    critic_opt = torch.optim.Adam(net.critic_params, CRITIC_LR)

    actor_tr  = [torch.zeros_like(p) for p in net.actor_params]
    critic_tr = [torch.zeros_like(p) for p in net.critic_params]

    for ep in trange(MAX_EPISODES, desc='episodes'):
        obs, _ = env.reset()
        ep_ret = 0.0
        # for z in actor_tr + critic_tr:
        #     z.zero_()

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0):
                env.render()

            # Interaction
            act, _ = net.act(obs)
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)
            obs_next, r, term, trunc, _ = env.step(act_clipped)
            done = term or trunc

            # Cast scalars to tensors early (consistent with #1 feedback)
            r_t         = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
            obs_t       = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            obs_next_t  = torch.as_tensor(obs_next, dtype=torch.float32, device=DEVICE)

            # Fresh value estimates
            with torch.no_grad():
                _, _, v_next = net(obs_next_t)
            _, _, v_pred_t = net(obs_t)  # current state value (used for both actor & critic)

            target = r_t + GAMMA * (0.0 if done else v_next)
            delta  = target - v_pred_t.detach()  # advantage, no grad

            # 1. Actor update (before critic)
            actor_opt.zero_grad()
            mu, std, _ = net(obs_t)  # trunk unchanged so far
            dist = Normal(mu, std)
            logp = dist.log_prob(torch.from_numpy(act).to(obs_t)).sum()
            logp.backward() # ∇ log π
            apply_traces(net.actor_params, actor_tr, scale=delta.item())  # uses current δ
            torch.nn.utils.clip_grad_norm_(net.actor_params, 0.5)
            actor_opt.step()

            # 2. Critic update (after actor)
            critic_opt.zero_grad()
            # re‑evaluate value because shared trunk is still unchanged (actor didn't touch it)
            _, _, v_pred_fresh = net(obs_t)
            critic_loss = 0.5 * (v_pred_fresh - target) ** 2
            critic_loss.backward()
            apply_traces(net.critic_params, critic_tr)
            torch.nn.utils.clip_grad_norm_(net.critic_params, 0.5)
            critic_opt.step()

            ep_ret += r
            if done:
                break
            obs = obs_next

        if ep % 10 == 0:
            print(f'Episode {ep:4d} | Return {ep_ret:7.1f}')

    env.close()

if __name__ == '__main__':
    train()
