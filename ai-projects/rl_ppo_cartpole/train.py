import gymnasium as gym, numpy as np, torch
import torch.nn as nn, torch.optim as optim
env=gym.make("CartPole-v1")
obs_dim=env.observation_space.shape[0]; act_dim=env.action_space.n
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.b=nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.pi=nn.Linear(64,act_dim); self.v=nn.Linear(64,1)
    def forward(self,x):
        h=self.b(x)
        return self.pi(h),self.v(h)
net=A(); opt=optim.Adam(net.parameters(),lr=3e-4)
gamma=0.99; lam=0.95; clip=0.2; steps=2048; epochs=10; minibatch=64
def roll():
    obs,info=env.reset(); buf=[]
    for _ in range(steps):
        x=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
        logits,v=net(x); p=torch.distributions.Categorical(logits=logits)
        a=p.sample().item(); nobs,r,term,trunc,_=env.step(a)
        buf.append((obs,a,r,term or trunc,p.log_prob(torch.tensor(a)).item(),v.item()))
        obs=nobs
    return buf
for _ in range(50):
    traj=roll()
    obs=np.array([t[0] for t in traj],np.float32)
    act=np.array([t[1] for t in traj],np.int64)
    rew=np.array([t[2] for t in traj],np.float32)
    done=np.array([t[3] for t in traj],np.bool_)
    logp=np.array([t[4] for t in traj],np.float32)
    val=np.array([t[5] for t in traj],np.float32)
    adv=np.zeros_like(rew); ret=np.zeros_like(rew); lastgaelam=0; nextv=0
    for t in reversed(range(len(rew))):
        mask=1.0- float(done[t])
        delta=rew[t]+gamma*nextv*mask - val[t]
        lastgaelam=delta+gamma*lam*mask*lastgaelam
        adv[t]=lastgaelam; nextv=val[t]
    ret=adv+val
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    idx=np.arange(len(rew))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0,len(rew),minibatch):
            j=idx[s:s+minibatch]
            x=torch.tensor(obs[j])
            logits,v=net(x)
            p=torch.distributions.Categorical(logits=logits)
            new_logp=p.log_prob(torch.tensor(act[j]))
            ratio=torch.exp(new_logp - torch.tensor(logp[j]))
            a=torch.tensor(adv[j],dtype=torch.float32)
            surr1=ratio*a; surr2=torch.clamp(ratio,1-clip,1+clip)*a
            polyloss=-(torch.min(surr1,surr2)).mean()
            vloss=(v.squeeze()-torch.tensor(ret[j])).pow(2).mean()
            ent=p.entropy().mean()
            loss=polyloss+0.5*vloss-0.01*ent
            opt.zero_grad(); loss.backward(); opt.step()
torch.save(net.state_dict(),"ppo_cartpole.pt")
