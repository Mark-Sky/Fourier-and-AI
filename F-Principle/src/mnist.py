from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.decomposition import PCA
from BasicFunc import mySaveFig, my_fft, highD_NUDFT, get_difference
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

LowFreqDrawAdd = 1e-5  # for plot. plot y+this number, in case of 0 in log-log
isShowPic = 0
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, projection_k):
    model.train()
    xs = []
    pred_ys = []
    ys = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, num_classes=10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        flatten_data = np.reshape(data.cpu().numpy(), (-1, 784))
        xs.append(flatten_data)
        ys.append(target_onehot.numpy())
        pred_ys.append(torch.exp(output).detach().cpu().numpy())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    all_xs = np.concatenate(xs, axis=0)  # (60000, 784)
    all_ys = np.concatenate(ys, axis=0)  # (60000, 10)
    all_pred_ys = np.concatenate(pred_ys, axis=0) # (60000, 10)


    ###########################################################
    ###########################################################
    
    """
    Calcuate and plot nonuniform discrete Fourier transform (NUDFT) using all_xs, all_ys and all_pred_ys
    You may refer to get_ft_multi() function in 
    https://github.com/xuzhiqin1990/F-Principle/blob/2b45d64390eabe502697c990232cbf4445a6c695/BasicFunc.py#L181
    You may need to modify this function.
    """
    train_size = all_xs.shape[0]
    k_index, fft_true_y = highD_NUDFT(all_xs, all_ys, projection_k, isnorm=False)
    k_index, fft_train_y = highD_NUDFT(all_xs, all_pred_ys, projection_k, isnorm=False)
    y1 = np.linalg.norm(fft_true_y, axis=1) / train_size
    y2 = np.linalg.norm(fft_train_y, axis=1) / train_size
    plt.figure()
    ax = plt.gca()
    plt.semilogy(k_index, y1 , 'r-', label='Trn_true')
    plt.semilogy(k_index, y2 , 'b--', label='Trn_fit')
    plt.xlabel('frequency', fontsize=18)
    plt.ylabel('amplitude', fontsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(fontsize=18)
    plt.title("High Dimension Fourier epoch%s" % epoch)
    ax.set_position(pos, which='both')
    fntmp = '../pics/high/Fourier/%s' % (epoch)
    mySaveFig(plt, fntmp, ax=ax, iseps=0)

    selected_index = [10, 30, 50, 70, 90]

    selected_k = k_index[selected_index]
    diff = get_difference(fft_train_y[selected_index]/ train_size, fft_true_y[selected_index]/ train_size)
    ###########################################################
    ###########################################################
    return diff

def test(model, device, test_loader):
    """ You will not need to change this.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    # Use GPU for faster training.
    # If CONVNet is too slow to you, you may use MLP instead.
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--model', type=str, default="MLP", choices=["MLP", "CONV"],
                        help='model (default: MLP)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    train_x = np.reshape(dataset1.data.numpy(), (-1, 784))
    train_y = dataset1.targets.numpy()

    pca = PCA(n_components=1)
    pca.fit(train_x)
    projection_k = torch.tensor(
        pca.components_, dtype=torch.float32, device=device)
    projection_k = pca.components_ # (1, 784)

    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.model == "MLP":
        model = MLPNet().to(device)
    else:
        model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    diffs = pd.DataFrame(columns=[i for i in range(1, args.epochs + 1)])


    for epoch in range(1, args.epochs + 1):
        diff = train(args, model, device, train_loader,
              optimizer, epoch, projection_k)
        diffs[epoch] = diff
        print(diff)
        test(model, device, test_loader)
        scheduler.step()

    sns.heatmap(data=diffs, cmap="RdBu_r")
    plt.show()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    st = time.time()
    main()
    et = time.time()
    print("time used {}".format(et - st))
