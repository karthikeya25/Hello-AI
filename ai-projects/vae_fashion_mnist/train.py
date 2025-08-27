import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device="cuda" if torch.cuda.is_available() else "cpu"
t=transforms.Compose([transforms.ToTensor()])
ds=datasets.FashionMNIST("data",train=True,download=True,transform=t)
dl=DataLoader(ds,batch_size=128,shuffle=True)
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Flatten(),nn.Linear(784,400),nn.ReLU())
        self.mu=nn.Linear(400,20); self.logvar=nn.Linear(400,20)
        self.dec=nn.Sequential(nn.Linear(20,400),nn.ReLU(),nn.Linear(400,784),nn.Sigmoid())
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h); logvar=self.logvar(h)
        std=torch.exp(0.5*logvar); eps=torch.randn_like(std); z=mu+eps*std
        xhat=self.dec(z).view(-1,1,28,28); return xhat,mu,logvar
m=VAE().to(device); opt=optim.Adam(m.parameters(),lr=1e-3)
def loss_fn(x,xhat,mu,logvar):
    bce=nn.functional.binary_cross_entropy(xhat,x,reduction="sum")
    kld=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return (bce+kld)/x.size(0)
for e in range(10):
    for x,_ in dl:
        x=x.to(device)
        xhat,mu,logvar=m(x)
        l=loss_fn(x,xhat,mu,logvar)
        opt.zero_grad(); l.backward(); opt.step()
torch.save(m.state_dict(),"vae_fashion_mnist.pt")
z=torch.randn(64,20).to(device)
gen=m.dec(z).view(-1,1,28,28).cpu()
torch.save(gen,"samples.pt")
