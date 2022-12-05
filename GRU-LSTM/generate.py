import math
import torch
import torch.nn as nn
from textloader import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO:  Ce fichier contient les différentes fonction de génération

soft = nn.LogSoftmax(dim=1)

def generate(rnn, emb, decoder, eos, start="", maxlen=200, LSTM=False, argmax=True):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    if start == "":
        h = None
        C = None
        generated = [torch.tensor(torch.randint(len(id2lettre), (1,))).to(device)]
        i = 0
        while generated[-1] != eos and i < maxlen:
            if LSTM:
                h, C = rnn.one_step(emb(generated[-1]), h, C)
            else:
                h = rnn.one_step(emb(generated[-1]), h)
            if argmax:
                generated.append(soft(decoder(h)).argmax(1))
            else:
                prob = Categorical(logits=soft(decoder(h)))
                generated.append(prob.sample())
            i += 1
        generated = torch.stack(generated)
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))

    else:
        h = None
        C = None
        if LSTM:
            h, C = rnn(emb(string2code(start).view(1, -1).to(device)).to(device))
            h = h[-1]
        else:
            h = rnn(emb(string2code(start).view(1, -1).to(device)).to(device))[-1]
        generated = [decoder(h).argmax(1)]
        i = 0
        while generated[-1] != eos and i < maxlen:
            if LSTM:
                h, C = rnn.one_step(emb(generated[-1]), h, C)
            else:
                h = rnn.one_step(emb(generated[-1]), h)
            if argmax:
                generated.append(soft(decoder(h)).argmax(1))
            else:
                prob = Categorical(logits=soft(decoder(h)))
                generated.append(prob.sample())
            i += 1
        generated = torch.stack(generated)
        print(start + "".join([id2lettre[int(i)] for i in generated.squeeze()]))


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200, LSTM=False, pNuc=None):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search
    h = None
    C = None
    if start == "":
        rnd = torch.randint(len(id2lettre), (1,))
        tmp = torch.tensor(rnd).to(device)

        start = id2lettre[int(rnd[0])]

        if LSTM:
            h, C = rnn.one_step(emb(tmp).float())
        else:
            h = rnn.one_step(emb(tmp).float())
    else:
        if LSTM:
            h, C = rnn(emb(string2code(start).view(1, -1).to(device)).float())
            h = h[-1]
        else:
            h = rnn(emb(string2code(start).view(1, -1).to(device)).float())[-1]

    candidats = soft(decoder(h)).squeeze()
    candidats = [([i], s, h, C) for i, s in enumerate(candidats.tolist())]
    candidats = sorted(candidats, key=lambda x: -x[1])[:k]

    for i in range(1, maxlen):
        currenttop = []
        for x in candidats:
            if LSTM:
                h, C = rnn.one_step(emb(torch.tensor(x[0][-1]).view(1).to(device)), x[2], x[3])
            else:
                h = rnn.one_step(emb(torch.tensor(x[0][-1]).view(1).to(device)), x[2])
            topk = soft(decoder(h)).squeeze()
            topk = [([i], s) for i, s in enumerate(topk.tolist())]
            currenttop += [(x[0] + j, x[1] + s, h, C) for j, s in topk]

        candidats = sorted(currenttop, key=lambda x: -x[1])[:k]
    best = candidats[0][0]

    if eos in best:
        end = best.index(eos)
        if end > 0:
            best = best[:best.index(eos) + 1]

    print(start + "".join([id2lettre[int(i)] for i in best]))