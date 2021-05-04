#reference : https://towardsdatascience.com/understanding-acgans-with-code-pytorch-2de35e05d3e4

import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from ACGAN_discriminator_1d import discriminator
from ACGAN_generator_1d import generator
from torch import optim
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

lr = 0.0001
epochs = 400
batch_size = 100
real_label = torch.FloatTensor(batch_size).cuda()
real_label.fill_(1)

fake_label = torch.FloatTensor(batch_size).cuda()
fake_label.fill_(0)

# last value of 'eval_noise_label' is not a value from normal dist. it's a temp.
eval_noise_label = torch.FloatTensor(batch_size, 7, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, 6))
#eval_label = np.random.randint(0, 7, batch_size)
eval_label = 6*np.ones(batch_size).astype('int64')
eval_onehot = np.zeros((batch_size, 7))
eval_onehot[np.arange(batch_size), eval_label] = 1
# print(eval_noise_[np.arange(batch_size), :7])
# print(eval_onehot[np.arange(batch_size)])
# eval_noise_[np.arange(batch_size), :7] = eval_onehot[np.arange(batch_size)]

eval_label = np.reshape(eval_label, (100,1))
eval_noise_label_ = np.concatenate((eval_noise_, eval_label), axis= 1)
eval_noise_label_ = (torch.from_numpy(eval_noise_label_))
eval_noise_label.data.copy_(eval_noise_label_.view(batch_size, 7, 1, 1))

eval_noise_label = eval_noise_label.cuda()
# print('eval_noise')
# print(eval_noise)

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




class Train_DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/train_freezed.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]

        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/test_freezed.csv',delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = Train_DiabetesDataset()
test_dataset = Test_DiabetesDataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)







gen = generator(7).cuda()
disc = discriminator().cuda()

gen.apply(weights_init)

optimD = optim.Adam(disc.parameters(), lr)
optimG = optim.Adam(gen.parameters(), lr)

source_obj = nn.BCELoss()  # source-loss

class_obj = nn.NLLLoss()  # class-loss

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader, 0):
        '''
        At first we will train the discriminator
        '''
        # training with real data----
        optimD.zero_grad()

        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs = scaler.transform(inputs)
        inputs = from_numpy(inputs)

        inputs, label = inputs.cuda().float(), labels.cuda().float()

        source_, class_ = disc(inputs)  # we feed the real images into the discriminator
        source_ = torch.reshape(source_, [-1])
        # print(source_.size())
        source_error = source_obj(source_, real_label)  # label for real images--1; for fake images--0

        label = torch.reshape(label, [-1]).cuda()
        class_error = class_obj(class_, label.long())
        error_real = source_error + class_error
        error_real.backward()
        optimD.step()

        accuracy = compute_acc(class_, label)  # getting the current classification accuracy

        # training with fake data now----

        noise_ = np.random.normal(0, 1,
                                  (batch_size, 6))  # generating noise by random sampling from a normal distribution

        label = np.random.randint(0, 7, batch_size)  # generating labels for the entire batch
        label_for_concat = np.reshape(label, (100,1))
        noise_label_ = np.concatenate((noise_, label_for_concat), axis = 1)
        noise_label = ((torch.from_numpy(noise_label_)).float())
        noise_label = noise_label.cuda()  # converting to tensors in order to work with pytorch

        label = ((torch.from_numpy(label)).long())
        label = label.cuda()  # converting to tensors in order to work with pytorch
        # ■■■■■■ Should be revised ■■■■■■
        noise_image = gen(noise_label)
        # ■■■■■■■■■■■■■■■■■■■■■■■■
        # print('noise image size')
        # print(noise_image.size())

        source_, class_ = disc(noise_image.detach())  # we will be using this tensor later on
        source_ = torch.reshape(source_, [-1])
        source_error = source_obj(source_, fake_label)  # label for real images--1; for fake images--0
        class_error = class_obj(class_, label)
        error_fake = source_error + class_error
        error_fake.backward()
        optimD.step()

        '''
        Now we train the generator as we have finished updating weights of the discriminator
        '''

        gen.zero_grad()
        source_, class_ = disc(noise_image)
        source_ = torch.reshape(source_, [-1])
        source_error = source_obj(source_,
                                  real_label)  # The generator tries to pass its images as real---so we pass the images as real to the cost function
        class_error = class_obj(class_, label)
        error_gen = source_error + class_error
        error_gen.backward()
        optimG.step()
        iteration_now = epoch * len(train_loader) + i

        print("Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch, epochs,
                                                                                                       error_fake,
                                                                                                       error_gen,
                                                                                                       accuracy))

        '''Saving the images by the epochs'''
        if i % epochs == 0:
            constructed = gen(eval_noise_label)
            constructed = (constructed.cuda()).detach().cpu().numpy()
            constructed = scaler.inverse_transform(constructed)


            constructed_fig = plt.plot([0,1,2,3,4,5,6,7,8,9],
                                       constructed[1,:],marker = '.',
                                       label="Constructed data")
            plt.legend(loc='upper right')
            plt.savefig('%s/constructed_fig %03d.png' % ('images/',epoch))
            plt.close()