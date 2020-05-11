import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # log_likelihood_bernoulli
    return torch.sum(target * torch.log(mu) + (torch.ones_like(target) - target) * torch.log(torch.ones_like(mu) - mu), dim=-1)


def log_likelihood_normal(mu, logvar, z):
    """
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """

    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    c = -mu.shape[-1] / 2 * torch.log(torch.tensor(np.full(shape=batch_size, fill_value=2 * math.pi, dtype=np.float32)))
    sigma_det = -logvar.sum(-1) / 2
    x1 = torch.matmul((z - mu).unsqueeze(1), torch.inverse(torch.diag_embed(torch.exp(logvar))))
    x = -1 / 2 * (x1.squeeze() * (z - mu)).sum(dim=1)
    return c + sigma_det + x


def log_mean_exp(y):
    """
    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # log_mean_exp
    a = torch.max(y, dim=1)[0].unsqueeze(-1)

    # return torch.logsumexp(y, 1) - math.log( sample_size)
    return torch.log((torch.exp(y - a)).sum(dim=-1)) - math.log(sample_size) + a.squeeze()


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    q = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, torch.diag_embed(torch.exp(logvar_q)))
    p = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, torch.diag_embed(torch.exp(logvar_p)))

    #kdl = torch.exp(q) * (q - p)
    kdl = torch.distributions.kl.kl_divergence(q, p)

    # kld
    return kdl


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    q = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, torch.diag_embed(torch.exp(logvar_q)))
    #p = log_likelihood_normal(mu_p, logvar_p)
    p = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, torch.diag_embed(torch.exp(logvar_p)))

    #kdl = torch.exp(q) * (q - p)
    kdl = torch.distributions.kl.kl_divergence(q, p)
    kdl_mean = kdl.mean(1)
    # kld
    return kdl_mean


mu = torch.rand((10, 5))
logvar = torch.rand((10, 5))
z = torch.rand((10, 5))

test_log_likelihood_normal = log_likelihood_normal(mu, logvar, z)

test_log_mean = log_mean_exp(z)
# print(test_log_mean)
# print(test_log_mean.shape)
test_kl = kl_gaussian_gaussian_analytic(mu, logvar, mu, logvar)
print(test_kl)

test_kl_mc = kl_gaussian_gaussian_mc(mu, logvar, mu, logvar, 1)
print(test_kl_mc)
