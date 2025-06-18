import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='turn on live viewer')
parser.add_argument('--render_every', type=int, default=1, help='draw 1 in N steps')
args = parser.parse_args()

ENV_ID      = 'BipedalWalker-v3'
GAMMA       = 0.99
LAMBDA      = 0.9
ENTROPY_COEF = 0.02
ACTOR_LR    = 1e-4
CRITIC_LR   = 3e-4
MAX_EPISODES = 5000
MAX_STEPS    = 1600
HIDDEN_SIZES = (128, 64)
DEVICE       = torch.device('cpu')

class RunningStats:
    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update_single(self, x):
        """Update with a single value"""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean - self.var) / self.count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

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
        act = TanhNormal(mu, std).rsample()
        return act.numpy(), v.item()

def init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            gain = np.sqrt(2)
            nn.init.orthogonal_(m.weight, gain)
            nn.init.constant_(m.bias, 0.)
    # small output layers
    nn.init.orthogonal_(net.mu_head.weight, 1e-2)
    nn.init.constant_(net.mu_head.bias, 0.)
    nn.init.orthogonal_(net.v_head.weight, 1e-2)
    nn.init.constant_(net.v_head.bias, 0.)

class TanhNormal:
    def __init__(self, mu, std):
        self.normal = Normal(mu, std)

    def rsample(self):
        z = self.normal.rsample() # re-parameterised noise
        return torch.tanh(z)

    def log_prob(self, a):
        # clamp to keep |a|<1 and stay in float32
        a = a.float().clamp_(-0.999, 0.999)
        z = torch.atanh(a)
        log_det = 2*(np.log(2) - z - torch.nn.functional.softplus(-2*z))
        return self.normal.log_prob(z) - log_det

class RunningNorm:
    def __init__(self, shape, eps=1e-4):
        self.mean = torch.zeros(shape, device=DEVICE)
        self.var  = torch.ones(shape,  device=DEVICE)
        self.count = eps

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var  = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot   = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def __call__(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

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
    init(net)

    # Two optimisers, shared trunk only in critic optimiser
    actor_opt  = torch.optim.Adam(net.actor_params,  ACTOR_LR)
    critic_opt = torch.optim.Adam(net.critic_params, CRITIC_LR)

    actor_tr  = [torch.zeros_like(p) for p in net.actor_params]
    critic_tr = [torch.zeros_like(p) for p in net.critic_params]

    obs_norm = RunningNorm(obs_dim)
    reward_stats = RunningStats()

    for ep in trange(MAX_EPISODES, desc='episodes'):
        obs, _ = env.reset()
        ep_ret = 0.0
        steps_survived = 0

        with torch.no_grad():
            net.logstd_head.data.mul_(0.999) # exponential decay
            net.logstd_head.data.clamp_(np.log(0.02), np.log(3.5))

        for z in actor_tr + critic_tr:
            z.zero_()

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0):
                env.render()

            # Interaction
            act, _ = net.act(obs)
            obs_next, r, term, trunc, _ = env.step(act)
            done = term or trunc

            # Update stats with this single reward
            reward_stats.update_single(r)

            # Normalize the reward for use in TD update
            r_normalized = reward_stats.normalize(r)

            # Cast scalars to tensors early
            r_t = torch.as_tensor(r_normalized, dtype=torch.float32, device=DEVICE)
            obs_t      = torch.as_tensor(obs , dtype=torch.float32, device=DEVICE)
            obs_norm.update(obs_t.unsqueeze(0))
            obs_t      = obs_norm(obs_t)

            obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=DEVICE)
            obs_next_t = obs_norm(obs_next_t)          # ***no second RunningNorm***

            # Fresh value estimates
            with torch.no_grad():
                _, _, v_next = net(obs_next_t)
            _, _, v_pred_t = net(obs_t)  # current state value (used for both actor & critic)

            target = r_t + GAMMA * (0.0 if done else v_next)
            delta  = target - v_pred_t.detach()  # advantage, no grad

            # 1 Actor update (before critic)
            mu, std, _ = net(obs_t)
            dist = TanhNormal(mu, std)
            act_t = torch.as_tensor(act, dtype=torch.float32, device=DEVICE)
            logp = dist.log_prob(act_t).sum()
            entropy = dist.normal.entropy().sum()
            actor_opt.zero_grad()
            action_penalty = 0.001 * (act_t ** 2).sum()
            actor_opt.zero_grad()
            (-logp - ENTROPY_COEF * entropy + action_penalty).backward()
            apply_traces(net.actor_params, actor_tr, scale=delta.detach())
            torch.nn.utils.clip_grad_norm_(net.actor_params, 0.5)
            actor_opt.step()

            # 2 Critic update (after actor)
            critic_opt.zero_grad()
            # Get value with current params (after actor update)
            _, _, v_pred = net(obs_t)  # Use full forward pass for consistency
            v_pred.backward()
            apply_traces(net.critic_params, critic_tr, scale=delta.detach())
            torch.nn.utils.clip_grad_norm_(net.critic_params, 0.5)
            critic_opt.step()

            ep_ret += r
            steps_survived += 1
            if done:
                break
            obs = obs_next

        if ep % 10 == 0:
            print(f'Episode {ep:4d} | Return {ep_ret:7.1f} | Steps {steps_survived:4d} | σ {net.logstd_head.exp().mean():.3f}')

    env.close()

if __name__ == '__main__':
    train()
