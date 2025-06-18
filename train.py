import argparse
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--render",        action="store_true", help="turn on live viewer")
parser.add_argument("--render_every",  type=int, default=1, help="draw 1 in N steps")
args = parser.parse_args()

device, ENV_ID = torch.device("cuda" if torch.cuda.is_available() else "cpu"), "BipedalWalker-v3"
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
    def forward(self, x):
        feat = self.shared(x)
        mu, v = self.mu_head(feat), self.v_head(feat).squeeze(-1)
        return mu, self.logstd_head.exp(), v
    @torch.no_grad()
    def act(self, obs):
        mu, std, v = self(torch.as_tensor(obs, dtype=torch.float32, device=device))
        act = Normal(mu, std).sample()
        return act.cpu().numpy(), v.cpu().item()

def update(trace, new_grad, γ, lam):          # eligibility-trace helper
    return γ * lam * trace + new_grad

# ------------------------------------------------------------------------------------
def train():
    render_mode = "human" if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim).to(device)
    actor_opt, critic_opt = torch.optim.Adam(net.parameters(), ACTOR_LR), torch.optim.Adam(net.parameters(), CRITIC_LR)
    actor_tr, critic_tr = [torch.zeros_like(p, device=device) for p in net.parameters()], \
                          [torch.zeros_like(p, device=device) for p in net.parameters()]

    for ep in trange(MAX_EPISODES, desc="episodes"):
        obs, _ = env.reset(seed=None)
        ep_ret = 0.0
        actor_tr[:] = [t.zero_() for t in actor_tr]; critic_tr[:] = [t.zero_() for t in critic_tr]

        for t in range(MAX_STEPS):
            if args.render and (t % args.render_every == 0): # draw only every N-th step
                env.render()

            act, v = net.act(obs)
            obs_next, r, term, trunc, _ = env.step(act)
            done = term or trunc
            with torch.no_grad():
                _, _, v_next = net(torch.as_tensor(obs_next, dtype=torch.float32, device=device))
                δ = r + GAMMA * (0.0 if done else v_next) - v         # TD-error

            # -- critic --------------------------------------------------------
            critic_opt.zero_grad()
            _, _, v_pred = net(torch.as_tensor(obs, dtype=torch.float32, device=device))
            v_pred.backward()
            for p, z in zip(net.parameters(), critic_tr):
                z.data = update(z.data, p.grad.data, GAMMA, LAMBDA)
                p.grad.data = -δ * z.data
            critic_opt.step()

            # -- actor ---------------------------------------------------------
            actor_opt.zero_grad()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mu, std, _ = net(obs_t); logp = Normal(mu, std).log_prob(torch.as_tensor(act, dtype=torch.float32, device=device)).sum()
            logp.backward()
            for p, z in zip(net.parameters(), actor_tr):
                z.data = update(z.data, p.grad.data, GAMMA, LAMBDA)
                p.grad.data = -δ * z.data           # ascent => minimise negative return
            actor_opt.step()

            ep_ret += r
            if done: break
            obs = obs_next

        if ep % 10 == 0:
            print(f"Episode {ep:4d} | Return {ep_ret:7.1f}")

    env.close()

if __name__ == "__main__":
    train()
