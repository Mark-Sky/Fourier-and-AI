import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from torch.optim import lr_scheduler
from BasicFunc import mySaveFig, my_fft
ComputeStepFFT = 60
LowFreqDrawAdd = 1e-5  # for plot. plot y+this number, in case of 0 in log-log
train_size = 120

isShowPic = 0
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out


def low_dim_data(num=train_size, period=5):
    """x in range[-10, 10]
    """
    def fn(x):
        res = np.zeros_like(x)
        sin_value = np.sin(x)
        cut = 0.6
        mask1 = sin_value > cut
        maskn1 = sin_value < -cut
        res[mask1] = 1
        res[maskn1] = -1
        return res
    x = np.linspace(-10, 10, num=num)
    _x = x / period * np.pi * 2
    y = fn(_x)
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    return x, y

def main():
    x, y = low_dim_data(num=train_size)
    fft_true_y = my_fft(y, ComputeStepFFT, x)

    model = MLP(
        input_size=1,
        output_size=1,
        hidden_sizes=[16, 128, 128, 16]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=50,
                                                           verbose=True)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    for ep in range(10000):
        _y = model(x)
        loss = torch.mean((y - _y)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_y = _y.detach().numpy()

        fft_train_y = my_fft(pred_y, ComputeStepFFT, x)
        y1 = fft_true_y / train_size
        y2 = fft_train_y / train_size
        #############################################
        #############################################
        
        """
        Plot spacial domain and Fourier domain.
        """
        # spacial domain
        if ep % 25 == 0:
            # spacial domain
            '''
            plt.figure()
            ax = plt.gca()
            plt.plot(x, pred_y, 'r-', label='output')
            plt.plot(x, y, 'g-', label='target')
            plt.legend(fontsize=16)
            plt.xlabel('x', fontsize=18)
            plt.ylabel('y', fontsize=18)
            plt.rc('xtick', labelsize=18)
            plt.rc('ytick', labelsize=18)
            plt.title('epoch: %s' % (ep), fontsize=18)
            fntmp = '../pics/low/spacial/%s' % (ep)
            mySaveFig(plt, fntmp, ax=ax, iseps=0)
            plt.show()
            '''
            # Fourier domain
            plt.figure()
            ax = plt.gca()
            plt.semilogy(y1 + LowFreqDrawAdd, 'r-', label='Trn_true')
            plt.semilogy(y2 + LowFreqDrawAdd, 'b--', label='Trn_fit')
            plt.xlabel('freq index', fontsize=18)
            plt.ylabel('|FFT|', fontsize=18)
            plt.rc('xtick', labelsize=18)
            plt.rc('ytick', labelsize=18)
            plt.legend(fontsize=18)
            plt.title("Fourier Low epoch%s" % ep)
            ax.set_position(pos, which='both')
            fntmp = '../pics/low/Fourier/%s' % (ep)
            mySaveFig(plt, fntmp, ax=ax, iseps=0)
        #############################################
        #############################################
            print(loss)



if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    main()