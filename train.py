import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange

# Use multiple CPU cores if available
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ----------------- Hyper‑parameters -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='turn on live viewer')
parser.add_argument('--render_every', type=int, default=1, help='draw 1 in N steps')
args = parser.parse_args()

ENV_ID = 'BipedalWalker-v3'
GAMMA = 0.99
LAMBDA = 0.95
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
MAX_EPISODES = 2000
MAX_STEPS = 1600
HIDDEN_SIZES = (128, 64)
DEVICE = torch.device('cpu')


# ----------------- Network -----------------
class ActorCritic(nn.Module):
    """Shared torso with separate heads; eligibility‑trace friendly."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        h1, h2 = HIDDEN_SIZES

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
        )

        # Actor head (mean + log‑std)
        self.mu_head = nn.Linear(h2, act_dim)
        self.logstd_head = nn.Parameter(torch.zeros(act_dim))

        # Critic head (scalar value)
        self.v_head = nn.Linear(h2, 1)

        # Parameter lists  ────────────────────────────────────────────
        # 1. shared trunk (belongs **only** to critic optimiser)
        self.shared_params: list[nn.Parameter] = list(self.shared.parameters())
        # 2. actor‑only params (mean + exploration scale)
        self.actor_params: list[nn.Parameter] = (
            list(self.mu_head.parameters()) + [self.logstd_head]
        )
        # 3. critic params include the shared trunk
        self.critic_params: list[nn.Parameter] = self.shared_params + list(
            self.v_head.parameters()
        )

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        mu = self.mu_head(feat)
        v = self.v_head(feat).squeeze(-1)
        std = self.logstd_head.exp()
        return mu, std, v

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        mu, std, v = self(obs_t)
        act = Normal(mu, std).sample()
        return act.cpu().numpy(), v.item()


# ----------------- Trace helper -----------------
def update(trace: torch.Tensor, new_grad: torch.Tensor, gamma: float, lam: float):
    """z ← γλ z + g_t"""
    return gamma * lam * trace + new_grad


# ----------------- Training loop -----------------
def train():
    render_mode = 'human' if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = ActorCritic(obs_dim, act_dim).to(DEVICE)

    # --- Optimisers ---
    #   *  critic_opt sees the shared trunk **once** (no duplication)
    #   *  actor_opt updates only actor‑specific parameters
    actor_opt = torch.optim.Adam(net.actor_params, ACTOR_LR)
    critic_opt = torch.optim.Adam(net.critic_params, CRITIC_LR)

    # Eligibility traces for each parameter set
    actor_tr = [torch.zeros_like(p) for p in net.actor_params]
    critic_tr = [torch.zeros_like(p) for p in net.critic_params]

    for ep in trange(MAX_EPISODES, desc='episodes'):
        obs, _ = env.reset()
        ep_ret = 0.0

        # reset traces at episode boundaries
        for z in actor_tr + critic_tr:
            z.zero_()

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0):
                env.render()

            # -------- Interaction --------
            act, v_pred = net.act(obs)
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)
            obs_next, r, term, trunc, _ = env.step(act_clipped)
            done = term or trunc

            # -------- Compute TD target --------
            with torch.no_grad():
                obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=DEVICE)
                _, _, v_next = net(obs_next_t)
                target = r + GAMMA * (0.0 if done else v_next)
                delta = target - v_pred  # scalar float

            # -------- Critic update --------
            critic_opt.zero_grad()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            _, _, v_pred_t = net(obs_t)
            critic_loss = 0.5 * (v_pred_t - target) ** 2
            critic_loss.backward()

            # Update critic traces
            with torch.no_grad():
                for p, z in zip(net.critic_params, critic_tr):
                    if p.grad is not None:
                        z.copy_(update(z, p.grad, GAMMA, LAMBDA))
                        p.grad.copy_(z)
            torch.nn.utils.clip_grad_norm_(net.critic_params, 0.5)
            critic_opt.step()

            # -------- Actor update --------
            actor_opt.zero_grad()
            mu, std, _ = net(obs_t)  # reuse obs_t
            dist = Normal(mu, std)
            logp = dist.log_prob(torch.from_numpy(act).to(obs_t)).sum()
            logp.backward()

            # Update actor traces (scaled by delta)
            with torch.no_grad():
                for p, z in zip(net.actor_params, actor_tr):
                    if p.grad is not None:
                        z.copy_(update(z, p.grad, GAMMA, LAMBDA))
                        p.grad.copy_(delta * z)
            torch.nn.utils.clip_grad_norm_(net.actor_params, 0.5)
            actor_opt.step()

            ep_ret += r
            if done:
                break
            obs = obs_next

        if ep % 10 == 0:
            print(f"Episode {ep:4d} | Return {ep_ret:7.1f}")

    env.close()


if __name__ == '__main__':
    train()
