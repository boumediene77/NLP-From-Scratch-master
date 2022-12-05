import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import datetime
from pathlib import Path

from torch.utils.tensorboard.summary import text

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]

#Taille du batch
BATCH_SIZE = 32
n = len(id2lettre)
nEmb = n//2

PATH = "data/"
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

with open(PATH+"trump_full_speech.txt", "r") as f:
    txt = f.read()
    ds = TrumpDataset(txt, maxsent = 100 ,maxlen=20)
    #data_train = TrumpDataset(txt[::len(txt)//2])
    #data_test = TrumpDataset(txt[len(txt)//2::])

data = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
#data_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
#data_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)

class State:
    def __init__(self, model, emb , optim):
        self.model = model
        self.emb = emb
        self.optimizer = optim
        self.epoch, self.iteration = 0, 0

def train_loop(dataloader,state=None):
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader): 
        one_hot_X = nn.functional.one_hot(X).float()
        Xemb = state.emb(one_hot_X)
        yhat = state.model(Xemb)
        decoded = state.model.decode(yhat).permute(1, 2, 0)
        L = nn.CrossEntropyLoss()
        loss = L(decoded , y)
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.iteration += 1
        train_loss += loss
    train_loss = train_loss / len(dataloader)
    return train_loss
    
def test_loop(dataloader, model , emb):
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            one_hot_X = nn.functional.one_hot(X).float()
            Xemb = emb(one_hot_X)
            yhat = model(Xemb)
            decoded = torch.stack([model.decode(i) for i in yhat],2)
            L = nn.CrossEntropyLoss()
            loss = L(decoded , y)
            test_loss += loss
    test_loss = test_loss / len(dataloader)
    return test_loss

def generate(state):
    seq = "i am goi"
    generated_text = seq
    for i in range(10):
        x = string2code(normalize(seq))
        x = torch.cat([torch.zeros(ds.MAX_LEN-x.size(0),dtype=torch.long),x])
        x = torch.stack([x])
        one_hot_X = nn.functional.one_hot(x.long(),num_classes=len(lettre2id)).float()
        x = state.emb(one_hot_X)
        yhat = state.model(x)
        decoded = state.model.decode(yhat)
        code = torch.argmax(decoded, dim=2)
        codee = code[-1]
        char = code2string(codee)
        seq += char
        seq = seq[1:]
        generated_text += char
    print(generated_text)


def train(data_train, save_path, Model, tensorboard_name, iterations=500):
    if save_path.is_file():
        with save_path.open('rb') as fp:
            state = torch.load(fp) 
            state.optimizer = torch.optim.Adam(list(state.model.parameters()) + list(state.emb.parameters()), lr=0.01)
    else :
        model = Model(nEmb, 100, n).to(device)
        emb = nn.Linear(n,nEmb).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()), lr=0.0001)
        state = State(model, emb , optimizer)
    for epoch in range(state.epoch, iterations):
        loss_train  = train_loop(data_train, state)
        with save_path.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)
        #loss_test = test_loop(data_test, state.model , state.emb)
        #writer.add_scalar(tensorboard_name+'/test', loss_test, epoch)
        writer.add_scalar(tensorboard_name+'/train', loss_train, epoch)
        if (epoch%100==0) : generate(state , "it is great")
        if (epoch%10==0) : print('Epoch:', epoch, 'Loss train: ',loss_train.item(), '\n\n')
        #print('Epoch:', epoch, '\n Loss test: ', loss_test, 'Loss train: ',loss_train, '\n\n')
    print("Done!")
    return state.model

savepath1 = Path('./trump_rnn.pt')
model1 = train(data, savepath1, RNN, "RNN_trump" ,iterations=1)


savepath = Path('./trump_rnn.pt')
if savepath.is_file():
    with savepath.open('rb') as fp:
        state = torch.load(fp, map_location=torch.device('cpu'))

generate(state)
