import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])

def pad_collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.
    max_length = len(max(samples,key = lambda p : len(p))) +1
    result = torch.zeros((len(samples),max_length),dtype=torch.float32)
    for i in range(len(samples)) :
        tmp = samples[i].tolist()
        tmp.append(EOS_IX)
        tmp = tmp + ([PAD_IX]* (max_length-len(tmp)))
        result[i] =torch.tensor(tmp,dtype=torch.float32)
    return result.T