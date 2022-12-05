import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from generate import *
from textloader import *

#  TODO:

class State(object) :
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    loss = CrossEntropyLoss(reduction='none')
    entropies = loss(output, target)
    mask = torch.where(target!=padcar, 1, 0)
    masked_output = mask.mul(entropies)
    return torch.mean(masked_output)

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, dim, latent, output):

        super(RNN, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output

        self.in_layer = nn.Linear((self.dim + self.latent), self.latent)
        self.out_layer = nn.Linear(self.latent, self.output)
        self.tanh = nn.Tanh()

    def one_step(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0],self.latent).to(device)
        temp = torch.cat((x, h), 1)
        return self.tanh(self.in_layer(temp))

    def forward(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0], self.latent).to(device)
        h_history = [h]
        for i in range(x.shape[1]):
            h_history.append(self.one_step(x[:, i, :], h_history[-1]))

        return h_history[1:]

    def decode(self, h):
        return self.out_layer(h)

class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim, latent, output):

        super(LSTM, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.i = nn.Linear((self.dim + self.latent), self.latent)
        self.f = nn.Linear((self.dim + self.latent), self.latent)
        self.o = nn.Linear((self.dim + self.latent), self.latent)
        self.in_layer = nn.Linear((self.dim + self.latent), self.latent)
        self.out_layer = nn.Linear(self.latent, self.output)

    def one_step(self, x, h=None, C= None):

        if h is None:
            h = torch.zeros(x.shape[0],self.latent).to(device)
        if C is None:
            C = torch.zeros(x.shape[0],self.latent).to(device)
        temp = torch.cat((x, h), 1)
        keep = self.sig(self.i(temp))
        forget = self.sig(self.f(temp))
        out = self.sig(self.o(temp))
        Ct = forget * C + keep * self.tanh(self.in_layer(temp))
        return out * self.tanh(Ct), Ct

    def forward(self, x, h=None, C=None):

        if h is None:
            h = torch.zeros(x.shape[0], self.latent).to(device)
        if C is None:
            C = torch.zeros(x.shape[0],self.latent).to(device)
        h_history = [h]
        for i in range(x.shape[1]):
            h, C = self.one_step(x[:, i, :], h_history[-1],C)
            h_history.append(h)

        return h_history[1:], C

    def decode(self, h):
        return self.out_layer(h)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        
        self.in_hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_hidden_layer_1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_hidden_layer_2 = nn.Linear(input_size + hidden_size, hidden_size)
            
        self.hidden_out_layer = nn.Linear(hidden_size, output_size)

    def one_step(self, x, h = None):
        if h is None:
          h = torch.zeros(x.shape[0],self.hidden_size).to(device)
        z = torch.sigmoid(self.in_hidden_layer_1(torch.cat((h, x), 1)))
        r = torch.sigmoid(self.in_hidden_layer_2(torch.cat((h, x), 1)))
        tmp = torch.tanh(self.in_hidden_layer(torch.cat((torch.mul(r, h), x), 1)))
        ht = torch.mul((1 - z), h) + torch.mul(z, tmp)
        return ht
    
    def forward(self, X):
        X = X.permute(1, 0, 2)
        h = torch.zeros(X.shape[1], self.hidden_size).to(device)
        # h_history = [h]
        h_history = torch.empty((len(X),X.shape[1], self.hidden_size), dtype=torch.float32).to(device)
        h_history[0] = h
        for i in range(len(X)):
            x = X[i]
            h = self.one_step(x, h)
            h_history[i] = h
        return h_history

    def decode(self,h):
        return self.hidden_out_layer(h) 



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

PATH = ""
BATCH_SIZE = 32
max_len = 30

with open(PATH + "trump_full_speech.txt", "r") as f:
    txt = f.read()

ds = TextDataset(txt, maxlen=max_len)
data = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, shuffle=True)

lr = 0.001
DIM_HIDDEN = 100
dimbed = 80
soft = nn.LogSoftmax(dim=1)


rnn = GRU(dimbed, DIM_HIDDEN, len(lettre2id)).to(device)
emb = nn.Embedding(len(id2lettre), dimbed, padding_idx=0).to(device)

def train_loop(dataloader,emb,state=None,LSTM = False):
  loss = maskedCrossEntropy
  losstrain = []
  for x in data:
      x = x.to(device)
      h = None
      C = None
      x = x.long()
      embed = emb(x) 
      l = 0
      for t in range(len(x) - 1):
          if LSTM:
              h, C = state.model.one_step(embed[t], h,C)
          else:
              h = state.model.one_step(embed[t], h)
          yhat = soft(rnn.decode(h))
          l+= loss(yhat,x[t+1],PAD_IX)
      l = l/(len(x) - 1)
      l.backward()
      state.optim.step()
      state.optim.zero_grad()
      state.iteration += 1
      losstrain.append(l)

  return torch.tensor(losstrain).mean()

def train(data_train, save_path, rnn , emb , tensorboard_name, iterations=1000 ,LSTM=False):
    if save_path.is_file():
        with save_path.open('rb') as fp:
            state = torch.load(fp) 
    else :
        optim = torch.optim.AdamW(rnn.parameters(), lr)
        state = State(rnn, optim)
    for epoch in range(state.epoch, iterations):
        losstrain = []
        l = train_loop(data_train,emb, state , LSTM)
        losstrain.append(l)
        with save_path.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)
        writer.add_scalar(tensorboard_name+'/train', torch.tensor(losstrain).mean(), epoch)
        generate(state.model, emb, state.model.decode, EOS_IX, start="the")
        if epoch % 10 == 0: print('Epoch:', epoch, 'Loss train: ',torch.tensor(losstrain).mean())
    print("Done!")
    return state.model

savepath1 = Path('./GRU_trump.pt')
model1 = train(data, savepath1, rnn, emb , "GRU_trump" ,iterations=1000)