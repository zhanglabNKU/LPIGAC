import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import minmax_scale

from models import GraphConv, AE, LP
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=144,                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between lncRNA space and protein space')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Hyperparameter beta')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)

def scaley(ymat):
    return (ymat-ymat.min())/ymat.max()

def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=5)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

mdi = np.loadtxt('interaction.txt',delimiter=',')
rnafeat = np.loadtxt('rnafeat.txt', delimiter=',')
rnafeat = minmax_scale(rnafeat,axis=0)

mdit = torch.from_numpy(mdi).float()
rnafeatorch = torch.from_numpy(rnafeat).float()
protfeat = np.loadtxt('protfeat.txt', delimiter=',')
protfeat = minmax_scale(protfeat,axis=0)
protfeatorch = torch.from_numpy(protfeat).float()
gm = norm_adj(rnafeat)
gd = norm_adj(protfeat)
if args.cuda:
    mdit = mdit.cuda()
    gm = gm.cuda()
    gd = gd.cuda()
    rnafeatorch = rnafeatorch.cuda()
    protfeatorch = protfeatorch.cuda()

class GNNp(nn.Module):
    def __init__(self):
        super(GNNp,self).__init__()
        self.gnnpl = LP(args.hidden,mdi.shape[1])
        self.gnnpd = LP(args.hidden,mdi.shape[0])

    def forward(self,y0):
        yl,zl = self.gnnpl(gm,y0)
        yd,zd = self.gnnpd(gd,y0.t())
        return yl,zl,yd,zd

def train(gnnp,y0,epoch,alpha):
    beta = args.beta
    optp = torch.optim.Adam(gnnp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    #optq = torch.optim.Adam(gnnq.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for e in range(epoch):
        gnnp.train()
        yl,zl,yd,zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl,y0)
        losspd = F.binary_cross_entropy(yd,y0.t())
        lossp = beta*(alpha*losspl + (1-alpha)*losspd) + F.mse_loss(torch.mm(zl,zd.t()),y0)
        optp.zero_grad()
        lossp.backward()
        optp.step()
        gnnp.eval()
        with torch.no_grad():
            yl,_,yd,_ = gnnp(y0)
        
        if e%20 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, lossp.item()))
        
    return alpha*yl+(1-alpha)*yd.t()

def trainres(A0):
    #gnnq = GNNq()
    gnnp = GNNp()
    if args.cuda:
        #gnnq = gnnq.cuda()
        gnnp = gnnp.cuda()

    train(gnnp,A0,args.epochs,args.alpha)
    #gnnq.eval()
    gnnp.eval()
    yli,_,ydi,_ = gnnp(A0)
    resi = args.alpha*yli + (1-args.alpha)*ydi.t()
    return resi

def fivefoldcv(A,alpha=0.5):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    res = torch.zeros(5,A.shape[0],A.shape[1])
    aurocl = np.zeros(5)
    auprl = np.zeros(5)
    for i in range(5):
        print("Fold {}".format(i+1))
        A0 = A.clone()
        for j in range(i*N//5,(i+1)*N//5):
            A0[idx[j],:] = torch.zeros(A.shape[1])
        
        resi = trainres(A0)
        #resi = scaley(resi)
        res[i] = resi
        
        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()
        
        auroc,aupr = show_auc(resi,args.data)
        aurocl[i] = auroc
        auprl[i] = aupr
        
    ymat = res[aurocl.argmax()]
    print("===Final result===")
    print('AUROC= %.4f +- %.4f | AUPR= %.4f +- %.4f' % (aurocl.mean(),aurocl.std(),auprl.mean(),auprl.std()))
    if args.cuda:
        return ymat.cpu().detach().numpy()
    else:
        return ymat.detach().numpy()

ymat = fivefoldcv(mdit,alpha=args.alpha)
ymat = scaley(ymat)
#np.savetxt(title+'.txt',ymat,fmt='%10.5f',delimiter=',')
np.savetxt('result_.txt',ymat,fmt='%10.5f',delimiter=',')
print("===Max result===")
show_auc(ymat,args.data)