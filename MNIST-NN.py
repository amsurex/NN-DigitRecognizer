# -*- coding: utf-8 -*-
"""MNIST.ipynb
Делала при помощи Google Colab
    https://colab.research.google.com/drive/1lltp5hIrFGjmhz7-5AEs-qsUsxqp9_Pv

## 1. Импортируем библиотеки
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import datasets
from torchvision import transforms

from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm

"""## 2. Загружаем данные"""

MNIST_train = datasets.MNIST('./mnist', train=True, download=True,
                             transform=transforms.ToTensor())

MNIST_test = datasets.MNIST('./mnist', train=False, download=True,
                            transform=transforms.ToTensor())

plt.imshow(MNIST_train[15][0].numpy()[0])

"""## 3. Создаем нейронную сеть"""

class Perceptron(torch.nn.Module):
    @property                     # для того чтобы использовать это как поле класса
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, input_dim=784, num_layers=0,     # p - вероятность dropout
                 hidden_dim=64, output_dim=10, p=0.0):
        super(Perceptron, self).__init__()

        self.layers = torch.nn.Sequential()          # задаем слои

        prev_size = input_dim
        for i in range(num_layers):
            self.layers.add_module('layer{}'.format(i),
                                  torch.nn.Linear(prev_size, hidden_dim))
            self.layers.add_module('relu{}'.format(i), torch.nn.ReLU())
            self.layers.add_module('dropout{}'.format(i), torch.nn.Dropout(p=p))
            prev_size = hidden_dim

        self.layers.add_module('classifier',
                               torch.nn.Linear(prev_size, output_dim))

    def forward(self, input):
        return self.layers(input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

model = Perceptron(num_layers=2) # bias - вектор, чтобы разд. гиперпл. не ч/з 0
model.to(device)

"""## 4. Тест, Обучение

#### функции
"""

def testing(model, dataset):
    generator = torch.utils.data.DataLoader(dataset, batch_size=64)

    pred = []
    real = []
    for x, y in generator:
        x = x.view([-1, 784]).to(device)
        y = y.to(device)

        pred.extend(torch.argmax(model(x), dim=-1).cpu().numpy().tolist())
        real.extend(y.cpu().numpy().tolist())

    return np.mean(np.array(real) == np.array(pred)), \
     classification_report (real, pred)

def trainer(model, dataset, loss_function, optimizer, epochs):
    for epoch in tqdm(range(epochs), leave=False):
        generator = torch.utils.data.DataLoader(dataset, batch_size=64,
                                              shuffle=True)
        for x, y in tqdm(generator, leave=False):
            optimizer.zero_grad()                                         # надо занулять градиент
            x = x.view([-1, 784]).to(device)                              # картинки в векторы
            y = y.to(device)

            output = model(x)
            loss = loss_function(output, y)
            loss.backward()                      # градиенты
            optimizer.step()                     # шаг оптим.

"""#### обучение"""

_ = model.train()
trainer(model=model,
        dataset=MNIST_train,
        loss_function=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        epochs=4)

"""#### тест"""

_ = model.eval()
acc, report = testing(model, MNIST_test)
print(report)

"""#### С помощью кросс-валидации подберем гиперпараметры"""

cross_val = KFold(3)
number_of_batch = cross_val.get_n_splits(MNIST_train)

grid = ParameterGrid({'num_layers': [0, 2],
                      'hidden_dim': [8, 64],
                      'p': [0.3, 0.7],
                      'lr': [0.001]})

X_train = MNIST_train.transform(MNIST_train.data.numpy()).transpose(0,1)
Y_train = MNIST_train.targets.data

scores = dict()
for item in tqdm(grid):
    list_of_scores = []
    for train_index, test_index in tqdm(cross_val.split(X_train),
                                        total=number_of_batch, leave=False):
        x_train_fold = X_train[train_index]
        x_test_fold = X_train[test_index]
        y_train_fold = Y_train[train_index]
        y_test_fold = Y_train[test_index]

        traindata = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        testdata = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)

        model = Perceptron(num_layers=item['num_layers'], p=item['p'],
                           hidden_dim=item['hidden_dim'])
        model.to(device)
        _ = model.train()
        trainer(model=model,
                dataset=traindata,
                loss_function=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=item['lr']),
                epochs=4)

        _ = model.eval()
        acc, report = testing(model, testdata)
        list_of_scores.append(acc)
    scores[str(item)] = [np.mean(list_of_scores)]

scores

"""#### Самое оптимальное решение: размерность скрытого слоя - 64, 'темп обучения - 0.001, количество слоев - 2, вероятность дропаута 0.3"""