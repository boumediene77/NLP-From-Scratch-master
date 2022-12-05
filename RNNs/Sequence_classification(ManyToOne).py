import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import RNN, device,SampleMetroDataset
from torch.utils.tensorboard import SummaryWriter
import datetime
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
from pathlib import Path
# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 64

PATH = ""


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
sample_train = SampleMetroDataset(matrix_train[:,:,:CLASSES,:DIM_INPUT],length=LENGTH)
sample_test = SampleMetroDataset(matrix_test[:,:,:CLASSES,:DIM_INPUT],length=LENGTH,stations_max=sample_train.stations_max)
data_train = DataLoader(sample_train, batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(sample_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optimizer = optim
        self.epoch, self.iteration = 0, 0

def train_loop(dataloader, state):
    size = len(dataloader.dataset)
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute loss
        output = state.model.forward(X)
        h_last = output[-1,:,:]
        out = state.model.decode(h_last)
        L = nn.CrossEntropyLoss()
        loss = L(out, y)
        # Backpropagation
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.iteration += 1
        train_loss += loss
        _, pred = torch.max(out.data, 1)
        train_acc += torch.sum( pred == y) / dataloader.batch_size
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss.item(), train_acc.item()

def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            output = model.forward(X)
            h_last = output[-1,:,:]
            out = model.decode(h_last)
            L = nn.CrossEntropyLoss()
            loss = L(out, y)
            test_loss += loss
            _, pred = torch.max(out.data, 1)
            test_acc += torch.sum( pred == y) / dataloader.batch_size
    test_acc = test_acc / num_batches
    test_loss = test_loss / num_batches
    return test_loss.item(), test_acc.item()

def train(data_train, data_test, save_path, Model, tensorboard_name, iterations=500):
    if save_path.is_file():
        with save_path.open('rb') as fp:
            state = torch.load(fp) 
    else :
        model = Model(DIM_INPUT, 10, CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        state = State(model, optimizer)
    for epoch in range(state.epoch, iterations):
        loss_train, acc_train = train_loop(data_train, state)
        with save_path.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)
        loss_test, acc_test = test_loop(data_test, state.model)
        writer.add_scalar(tensorboard_name+'/train', loss_test, epoch)
        writer.add_scalar(tensorboard_name+'/test', loss_train, epoch)
        print('Epoch:', epoch, '\n Loss test: ', loss_test, 'Loss train: ',loss_train, '\nAcc test: ',acc_test, ' Acc train: ', acc_train, '\n\n')
    print("Done!")
    return state.model

savepath1 = Path('./rnn.pt')
model1 = train(data_train,data_test, savepath1, RNN, "RNN" ,iterations=1000)


'''
for batch, (X, y) in enumerate(data_train):
    output = rnn.forward(X)
    h_last = output[-1,:,:]
    out = rnn.decode(h_last)
    print(out.size(), y.size())
    loss = nn.CrossEntropyLoss()
    output = loss(out, y)
    print(output)
    break
'''