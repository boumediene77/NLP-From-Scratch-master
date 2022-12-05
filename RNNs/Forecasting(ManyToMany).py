from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path


# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "data/"
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
class State:
    def __init__(self, model, optim):
        self.model = model
        self.optimizer = optim
        self.epoch, self.iteration = 0, 0

def train_loop(dataloader, state):
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.view ((X.shape[0], 19, -1))
        y = y.view ((X.shape[0], 19, -1))
        output = state.model.forward(X)
        out = state.model.decode(output) 
        out = out.permute(1, 0, 2)
        L = nn.MSELoss() 
        loss = L(out,y)
        # Backpropagation
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.iteration += 1
        train_loss += loss

    train_loss = train_loss / len(dataloader)
    return train_loss

def test_loop(dataloader, model):
    num_batches = len(dataloader)
    test_loss= 0
    with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X = X.view ((X.shape[0], 19, -1))
                y = y.view ((X.shape[0], 19, -1))
                output = model.forward(X)
                out = model.decode(output)
                out = out.permute(1, 0, 2)
                L = nn.MSELoss() 
                loss = L(out,y)
                test_loss += loss
    test_loss = test_loss / num_batches
    return test_loss


def train(data_train, data_test, save_path, Model, tensorboard_name, iterations=500):
    if save_path.is_file():
        with save_path.open('rb') as fp:
            state = torch.load(fp) 
    else :
        model = Model(DIM_INPUT * CLASSES, 10, DIM_INPUT * CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        state = State(model, optimizer)
    for epoch in range(state.epoch, iterations):
        loss_train  = train_loop(data_train, state)
        with save_path.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)
        loss_test = test_loop(data_test, state.model)
        writer.add_scalar(tensorboard_name+'/test', loss_test, epoch)
        writer.add_scalar(tensorboard_name+'/train', loss_train, epoch)
        print('Epoch:', epoch, '\n Loss test: ', loss_test, 'Loss train: ',loss_train, '\n\n')
    print("Done!")
    return state.model

savepath1 = Path('./rnn_pred.pt')
model1 = train(data_train,data_test, savepath1, RNN, "RNN_pred" ,iterations=1000)
X,y = next(iter(data_test))
X = X.view ((X.shape[0], 19, -1))
y = y.view ((X.shape[0], 19, -1))
output = model1.forward(X)
out = model1.decode(output)
out = out.permute(1, 0, 2)
L = nn.MSELoss() 
loss = L(out,y)
for i in range(10):
    for j in range (19):
        print("prediction was " + str(out[i][j].tolist())+ " real was " +str(y[i][j].tolist()))
        print()
print(loss)