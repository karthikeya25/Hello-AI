import torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
t=transforms.Compose([transforms.Resize(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train=datasets.CIFAR10(root="data",train=True,download=True,transform=t)
test=datasets.CIFAR10(root="data",train=False,download=True,transform=t)
tr=DataLoader(train,batch_size=128,shuffle=True,num_workers=2)
te=DataLoader(test,batch_size=256,shuffle=False,num_workers=2)
device="cuda" if torch.cuda.is_available() else "cpu"
m=timm.create_model("vit_tiny_patch16_224",pretrained=True,num_classes=10).to(device)
opt=optim.AdamW(m.parameters(),lr=3e-4)
sc=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
ce=nn.CrossEntropyLoss()
for e in range(10):
    m.train()
    for x,y in tr:
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        l=ce(m(x),y); l.backward(); opt.step()
    sc.step()
m.eval(); correct=0; total=0
with torch.no_grad():
    for x,y in te:
        x,y=x.to(device),y.to(device)
        p=m(x).argmax(1)
        correct+=(p==y).sum().item(); total+=y.size(0)
acc=correct/total
torch.save(m.state_dict(),"vit_cifar10.pt")
print(acc)
