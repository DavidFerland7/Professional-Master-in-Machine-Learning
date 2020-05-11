import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
from sampler import svhn_sampler
from model import Critic, Generator
from torch import optim
import torch.nn.functional as F
from torch import autograd
import os
import scipy.misc
from scipy.misc import imsave
from datetime import datetime


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples / rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)


def vf_wasserstein_distance(x, y, critic):
    """
    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    f_x = critic(x)
    f_y = critic(y)
    mean_f_x = f_x.mean()
    mean_f_y = f_y.mean()

    return mean_f_x - mean_f_y


def lp_reg(x, y, critic):
    """
    The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm.

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    eps = torch.rand(x.size()[0], device=x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    eps = eps.expand(x.size())

    x_hat = x * eps + y * (1 - eps)
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)

    f_x_hat = critic(x_hat)

    x_hat_grad = torch.autograd.grad(
        outputs=f_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(f_x_hat.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    return (F.relu((x_hat_grad.view(x.shape[0], -1).norm(2, dim=1) - 1)) ** 2).mean()


if __name__ == '__main__':
    time_now = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    data_root = './GAN/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    n_iter = 50000  # N training iterations
    n_critic_updates = 5  # N critic updates per generator update
    lp_coeff = 10  # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    log_every = 500
    save_images_every = 5000

    train_loader, _, _ = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # Define dataloader
    dataloader_iter = iter(cycle(train_loader))

    ### TRAINING LOOP ###
    loss_critic_cum = 0
    loss_generator_cum = 0
    for i in range(n_iter * n_critic_updates):

        ########### UPDATE CRITIC - every 1 iteration ###########

        # Init
        critic.zero_grad()
        # turn back on gradients computation for critic
        for param in critic.parameters():
            param.requires_grad = True

        # Get True sample
        real_imgs, _ = next(dataloader_iter)
        # try:
        #     # Samples the batch
        #     real_imgs, _ = next(dataloader_iter)
        # except StopIteration:
        #     # restart the generator if the previous generator is exhausted.
        #     dataloader_iter = iter(train_loader)
        #     real_imgs, _ = next(dataloader_iter)

        real_imgs = autograd.Variable(real_imgs).to(device)

        ### Generate fake images ###
        z = torch.randn(train_batch_size, z_dim, device=device)
        z = autograd.Variable(z, requires_grad=False)
        fake_imgs = autograd.Variable(generator(z).data)

        # compute loss
        dist = vf_wasserstein_distance(real_imgs, fake_imgs, critic)
        reg = lp_coeff * lp_reg(real_imgs, fake_imgs, critic)
        loss_critic = -dist + reg  # maximize real-fake
        # save cum for print
        loss_critic_cum += loss_critic.data.item()

        # Backprop
        loss_critic.backward()

        # Update parameters
        optim_critic.step()

        ### print loss and reinitialize cum loss ###
        if i % log_every == (log_every - 1):
            print("(iter+1)= {}  || avg (over {}) critic loss= {}".format(i + 1, log_every, loss_critic_cum / log_every))
            loss_critic_cum = 0

        ########### UPDATE GENERATOR - every 'n_critic_updates' iterations ###########
        if i % n_critic_updates == (n_critic_updates - 1):
            # Init
            generator.zero_grad()
            # turn off gradients computations for critic
            for param in critic.parameters():
                param.requires_grad = False

            ### Generate fake images ###
            z = torch.randn(train_batch_size, z_dim, device=device)
            z = autograd.Variable(z)
            fake_imgs = generator(z)

            # compute loss
            loss_generator = -critic(fake_imgs).mean()  # minimize fake
            # save cum for print
            loss_generator_cum += loss_generator.data.item()
            # Backprop
            loss_generator.backward()

            # Update parameters
            optim_generator.step()

            if i % log_every == (log_every - 1):
                print("(iter+1)= {}  || avg (over {}) generator loss= {}".format(i + 1, int(log_every / 5), (loss_generator_cum / (log_every / 5))))
                loss_generator_cum = 0

        ########### SAVE IMGS TO DISK - every 'save_images_every' iterations ###########
        if i % save_images_every == (save_images_every - 1):
            z = torch.randn(train_batch_size, z_dim, device=device)
            z = autograd.Variable(z, requires_grad=False)

            samples = generator(z)
            samples = samples.view(train_batch_size, 3, 32, 32)
            # print samples.size()

            samples = samples.cpu().data.numpy()

            # create directory if not exist
            os.makedirs(data_root + 'output_{}/'.format(time_now), exist_ok=True)

            save_images(
                samples,
                data_root + 'output_{}/samples_{}.png'.format(time_now, i + 1)
            )

    # save models
    torch.save(generator, data_root + 'output_{}/generator.pt'.format(time_now))
    torch.save(critic, data_root + 'output_{}/critic.pt'.format(time_now))
