import torch, argparse
from torch.nn import functional as F
from torch import nn
import numpy as np
import math


class CovariateVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    """

    def __init__(self, p):
        super(CovariateVAE, self).__init__()

        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 100)
        self.fc21 = nn.Linear(100, 2)
        self.fc22 = nn.Linear(100, 2)
        self.fc3 = nn.Linear(2, 100)
        self.fc31 = nn.Linear(100, 40)
        self.fc4 = nn.Linear(40, 4)
        self.p = p
        self.dropout1 = nn.Dropout(p=self.p)
        self.dropout2 = nn.Dropout(p=self.p)
        self.dropout3 = nn.Dropout(p=self.p)

        self.dropout4 = nn.Dropout(p=self.p)
        self.dropout5 = nn.Dropout(p=self.p)
        self.dropout6 = nn.Dropout(p=self.p)

        self.log_scale = nn.Parameter(torch.Tensor([0.0, 0.0, 0.0, 0.0]))

    def encode(self, x):
        h1 = self.dropout1(F.relu(self.fc1(x)))
        h2 = self.dropout2(F.relu(self.fc2(h1)))
        return self.fc21(h2), self.fc22(h2)

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.dropout4(F.relu(self.fc3(z)))
        h5 = self.dropout6(F.relu(self.fc31(h3)))
        return self.fc4(h5)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4))
        z = self.sample_latent(mu, logvar)
        return self.decode(z), mu, logvar

    def gaussian_likelihood(self, x_hat, x, mask):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = torch.mul(dist.log_prob(x), mask.view(-1, 4))
        return log_pxz.sum(-1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, recon_x, x, mask):
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, 4), x.view(-1, 4)), mask.view(-1, 4))

        mask_sum = torch.sum(mask.view(-1, 4), dim=1)
        mask_sum = mask_sum.unsqueeze(1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        mse = torch.mean(mse)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return mse

class ConvVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False, X_dim=0):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim
        self.X_dim = X_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 9 * 9 + self.X_dim, 300)
        self.fc21 = nn.Linear(300, 30)
        self.fc211 = nn.Linear(30, self.latent_dim)
        self.fc221 = nn.Linear(30, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(self.latent_dim + self.X_dim, 30)
        self.fc31 = nn.Linear(30, 300)
        self.fc4 = nn.Linear(300, 32 * 9 * 9)

        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)
        
        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

        self.fc1_X = nn.Linear(self.X_dim, 40)
        self.fc2_X = nn.Linear(40, 20)
        self.fc21_X = nn.Linear(20, self.X_dim)
        self.fc22_X = nn.Linear(20, self.X_dim)

        self.log_scale = nn.Parameter(torch.zeros(self.num_dim))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def gaussian_likelihood(self, x_hat, x, mask):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = torch.mul(dist.log_prob(x), mask)
        return log_pxz.sum(-1)

    def encode(self, x, X_cov):
        """
        Encode the passed parameter

        :param x: input data
        :param X_cov: covariates
        :return: variational mean and variance
        """
        # convolution
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)

        # MLP
        z = z.view(-1, 32 * 9 * 9)
        X_cov = X_cov.view(-1, self.X_dim)

        h1 = F.relu(self.fc1(torch.cat((z, X_cov), 1)))
        h2 = F.relu(self.fc21(h1))

        h1_X = F.relu(self.fc1_X(X_cov))
        h2_X = F.relu(self.fc2_X(h1_X))

        return self.fc211(h2), self.fc221(h2), self.fc21_X(h2_X), self.fc22_X(h2_X)

    def decode(self, z, X_cov):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        z = z.view(-1, self.latent_dim)
        X_cov = X_cov.view(-1, self.X_dim)
        # MLP
        x = F.relu(self.fc3(torch.cat((z, X_cov), 1)))
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = x.view(-1, 32, 9, 9)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, X_cov, X_cov_mask):
        mu, log_var, mu_X, log_var_X = self.encode(x, X_cov)
        z = self.sample_latent(mu, log_var)
        X_tilde = self.sample_latent(mu_X, log_var_X)
        z_X = X_tilde * (1 - X_cov_mask) + X_cov * X_cov_mask
        return self.decode(z, z_X), mu, log_var, mu_X, log_var_X, X_tilde

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        recon_x = recon_x.view(-1, self.num_dim)
        x = x.view(-1, self.num_dim)
        mask = mask.view(-1, self.num_dim)

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


class SimpleVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        self.fc1 = nn.Linear(num_dim, 300)
        self.fc21 = nn.Linear(300, 30)
        self.fc211 = nn.Linear(30, latent_dim)
        self.fc221 = nn.Linear(30, latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.fc31 = nn.Linear(30, 300)
        self.fc4 = nn.Linear(300, num_dim)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc21(h1))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h4))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.num_dim))
        z = self.sample_latent(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)
