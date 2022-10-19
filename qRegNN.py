from imports import *

#original version via Sam Bright-Thonney Oct 19 2022

class qreg(nn.Module):
    def __init__(self,input_dim,quantiles,widths=[30,30,30]):
        super().__init__()
        self.quantiles = quantiles
        out_dim = len(self.quantiles)
        layers = [nn.Linear(input_dim,widths[0]),
                  nn.Tanh()]
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(widths[-1],out_dim))
        self.transform = nn.Sequential(*layers)

    def forward(self,x):
        return self.transform(x)
    
def qLoss(Ypred,Ytrue,quantiles):
    qcast = quantiles.reshape(1,-1).repeat(Ytrue.shape[0],1)
    err = Ytrue - Ypred
    loss = torch.mean(torch.maximum(qcast*err,(qcast-1)*err),dim=1)
    loss = torch.mean(loss)
    del err
    return loss

def train(model,Xtrain,Ytrain,n_epoch=1000,lr=1e-4,check=100):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    losses = []
    for i in range(n_epoch):
        Ypred = model(Xtrain)
        if i==0:
            print(Xtrain.shape)
            print(Ytrain.shape)
            print(Ypred.shape)
            print(model.quantiles.reshape(1,-1).repeat(Ytrain.shape[0],1).shape)
        loss = qLoss(Ypred,Ytrain,model.quantiles)
        losses.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del Ypred,loss
        if (i+1)%check == 0:
            print("Epoch {}, Loss: {}".format(i+1,losses[-1]))
    plt.plot(np.arange(n_epoch),losses)
    
def train_gen(model,X,Y,bs=10000,n_epoch=1000,lr=1e-4,check=100):
    gen = utils.DataLoader(torch.cat((X,Y),dim=1),batch_size=10000,shuffle=True,generator=torch.Generator(device='cuda'))
    #gen = myLoader(torch.cat((X,Y),dim=1),bs=bs)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    losses = []
    for i in range(n_epoch):
        min_loss = 9999999
        #for j,xy in enumerate(gen.serve()):
        for j,xy in enumerate(gen):
            split = xy.shape[1]-len(model.quantiles)
            Xtrain = xy[:,:split]
            Ytrain = xy[:,split:]
            Ypred = model(Xtrain)
            if i==0 and j==0:
                print(Xtrain.shape)
                print(Ytrain.shape)
                print(Ypred.shape)
                print(model.quantiles.reshape(1,-1).repeat(Ytrain.shape[0],1).shape)
            loss = qLoss(Ypred,Ytrain,model.quantiles)
            if loss.detach().cpu().numpy() < min_loss:
                min_loss = loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del Ypred,loss
        losses.append(min_loss)
        if (i+1)%check == 0:
            print("Epoch {}, Loss: {}".format(i+1,losses[-1]))
    plt.plot(np.arange(n_epoch),losses)

class myLoader:
    def __init__(self,data,bs=10000):
        self.data = data
        self.bs = bs
        self.n = data.shape[0]
    def serve(self):
        perm = torch.randperm(self.n)
        return torch.split(self.data[perm],self.bs)
