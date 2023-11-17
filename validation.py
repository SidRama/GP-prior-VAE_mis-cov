from torch import nn
from torch.utils.data import DataLoader
import torch
from elbo_functions import deviance_upper_bound, elbo, minibatch_sgd
from utils import batch_predict_varying_T


def kl_divergence_simple(z, mu, std, covariates_mask):
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
    kl = kl * (1 - covariates_mask)
    kl = kl.sum()
    return kl


def validation_dubo(latent_dim, covar_module0, covar_module1, likelihood, train_xt, m, log_v, z, P, T, eps):
    """
    Efficient KL divergence using the variational mean and variance instead of a sample from the latent space (DUBO).
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.exp(log_v)
    torch_dtype = torch.double
    x_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[1], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[1], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(stacked_x_st, stacked_x_st).evaluate().transpose(0,1)
    B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + torch.eye(T, dtype=torch.double).to(device) * likelihood.noise_covar.noise.unsqueeze(dim=2)).transpose(0,1)
    LB_st = torch.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device), LB_st)

    dubo_sum = torch.tensor([0.0]).double().to(device)
    for i in range(latent_dim):
        m_st = torch.reshape(m[:, i], [P, T, 1]).to(device)
        v_st = torch.reshape(v[:, i], [P, T]).to(device)
        K0xz_st = torch.reshape(K0xz[i], [P, T, K0xz.shape[2]]).to(device)
        iB_K0xz = torch.matmul(iB_st[i], K0xz_st).to(device)
        K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz[i], 0, 1), torch.reshape(iB_K0xz, [P*T, K0xz.shape[2]])).to(device)
        W = K0zz[i] + K0zx_iB_K0xz
        W = (W + W.T) / 2
        LW = torch.cholesky(W).to(device)
        logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz[i]))).to(device)
        logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st[i], dim1=-2, dim2=-1))).to(device)
        logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
        logDetSigma = -logDetK0zz + logDetB + logDetW
        iB_m_st = torch.solve(m_st, B_st[i])[0].to(device)
        qF1 = torch.sum(m_st*iB_m_st).to(device)
        p = torch.matmul(K0xz[i].T, torch.reshape(iB_m_st, [P * T])).to(device)
        qF2 = torch.sum(torch.triangular_solve(p[:,None], LW, upper=False)[0] ** 2).to(device)
        qF = qF1 - qF2
        tr = torch.sum(iB_st[i] * K0_st[i]) - torch.sum(K0zx_iB_K0xz * iK0zz[i])
        logDetD = torch.sum(torch.log(v[:, i])).to(device)
        tr_iB_D = torch.sum(torch.diagonal(iB_st[i], dim1=-2, dim2=-1)*v_st).to(device)
        D05_iB_K0xz = torch.reshape(iB_K0xz*torch.sqrt(v_st)[:,:,None], [P*T, K0xz.shape[2]])
        K0zx_iB_D_iB_K0zx = torch.matmul(torch.transpose(D05_iB_K0xz,0,1), D05_iB_K0xz).to(device)
        tr_iB_K0xz_iW_K0zx_iB_D = torch.sum(torch.diagonal(torch.cholesky_solve(K0zx_iB_D_iB_K0zx, LW))).to(device)
        tr_iSigma_D = tr_iB_D - tr_iB_K0xz_iW_K0zx_iB_D
        dubo = 0.5*(tr_iSigma_D + qF - P*T + logDetSigma - logDetD + tr)
        dubo_sum = dubo_sum + dubo
    return dubo_sum

def validate_simple_batch(nnet_model, type_nnet, dataset, type_KL, num_samples, latent_dim, covar_module0, likelihoods,
                          zt_list, weight, train_mu, train_x, loss_function, batch_size, m, H, natural_gradient, eps=1e-6):

    """
    Perform validation given the validation set.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """
    print("Testing the model with a validation set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = train_x.shape[0]
    n_batches = (N + batch_size - 1) // (batch_size)

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    Q = train_x.shape[1]

    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_log_var = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_labels = torch.zeros(len(dataset), Q, dtype=torch.double, requires_grad=False).to(device)

    recon_loss_sum = 0
    nll_loss_sum = 0
    kld_loss_sum = 0
    net_loss_sum = 0
    iid_kld_sum = 0

    for batch_idx, sample_batched in enumerate(dataloader):
        data = sample_batched['digit'].double().to(device)
        train_x = sample_batched['label'].double().to(device)
        mask = sample_batched['mask'].double().to(device)
        N_batch = data.shape[0]

        data = data * mask.reshape(N_batch, 1, 36, 36)
        label_mask = sample_batched['label_mask'].double()
        label_mask = label_mask.to(device)
        covariates_mask = label_mask[:, 0:4]

        covariates = train_x[:, 2:6] * covariates_mask

        train_label_means = torch.sum(covariates, dim=0) / torch.sum(covariates_mask, dim=0)
        covariates_mask_bool = torch.gt(covariates_mask, 0)
        train_label_std = torch.zeros(covariates.shape[1])
        for col in range(0, covariates.shape[1]):
            train_label_std[col] = torch.std(covariates[covariates_mask_bool[:, col], col])

        covariates_norm = (covariates - train_label_means) / train_label_std

        noise_replace = torch.zeros_like(covariates_norm)

        covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

        recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, covariates_mask)
        X_tilde_norm = X_tilde * (1 - covariates_mask) + covariates_norm * covariates_mask
        X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
        X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates * covariates_mask

        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(X_tilde_norm.view(-1, covariates.shape[1]), covariates_norm.view(-1, covariates.shape[1])),
                       covariates_mask.view(-1, covariates.shape[1]))
        mask_sum = torch.sum(covariates_mask.view(-1, covariates.shape[1]), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse_X = torch.sum(torch.sum(se, dim=1) / mask_sum)

        [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
        recon_loss = torch.sum(recon_loss)
        nll_loss = torch.mean(nll)

        gp_loss_avg = torch.tensor([0.0]).to(device)
        net_loss = torch.tensor([0.0]).to(device)
        penalty_term = torch.tensor([0.0]).to(device)

        # sample X_hat from q_X
        std_X = torch.exp(log_var_X / 2)
        q_X = torch.distributions.Normal(mu_X, std_X)
        X_hat_sample = q_X.rsample()

        kl_X = kl_divergence_simple(X_tilde_norm, mu_X, std_X, covariates_mask)

        PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
        kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, X_hat, mu, log_var,
                                                 zt_list, N_batch, natural_gradient, eps, N)

        recon_loss = recon_loss * N / N_batch
        nll_loss = nll_loss * N / N_batch

        if loss_function == 'nll':
            net_loss = nll_loss + kld_loss + kl_X
        elif loss_function == 'mse':
            kld_loss = kld_loss / latent_dim
            net_loss = recon_loss + mse_X + weight * (kld_loss + kl_X)

        net_loss_sum += net_loss.item() / n_batches
        recon_loss_sum += recon_loss.item() / n_batches
        nll_loss_sum += nll_loss.item() / n_batches
        kld_loss_sum += kld_loss.item() / n_batches

        print('Validation set - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
            net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum))

    return recon_loss_sum

def validate(nnet_model, type_nnet, dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods,
             zt_list, T, weight, train_mu, train_x, id_covariate, loss_function, eps=1e-6):
    """
    Obtain KL divergence of validation set.

    :param nnet_model: neural network model
    :param type_nnet: type of encoder/decoder
    :param dataset: dataset to use
    :param type_KL: type of KL divergence computation
    :param num_samples: number of samples
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param zt_list: list of inducing points
    :param T: number of timepoints
    :param weight: value for the weight
    :param train_mu: mean on training set
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param eps: jitter
    :return: KL divergence between variational distribution
    """
    print("Testing the model with a validation set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = T
    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    Q = len(dataset[0]['label'])
    P = len(dataset) // T

    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_log_var = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_labels = torch.zeros(len(dataset), Q, dtype=torch.double, requires_grad=False).to(device)

    recon_loss_sum = 0
    nll_loss_sum = 0
    for batch_idx, sample_batched in enumerate(dataloader):
        indices = sample_batched['idx']
        data = sample_batched['digit'].double().to(device)
        mask = sample_batched['mask'].double().to(device)
        full_labels[indices] = sample_batched['label'].double().to(device)

        covariates = torch.cat((full_labels[indices, :id_covariate], full_labels[indices, id_covariate+1:]), dim=1)
        if type_nnet == 'rnn':
            recon_batch, mu, log_var = nnet_model(data, covariates)
        else:
            recon_batch, mu, log_var = nnet_model(data)

        full_mu[indices] = mu
        full_log_var[indices] = log_var

        [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
        recon_loss = torch.sum(recon_loss)
        nll = torch.sum(nll)

        recon_loss_sum = recon_loss_sum + recon_loss.item()
        nll_loss_sum = nll_loss_sum + nll.item()

    gp_losses = 0
    gp_loss_sum = 0
    param_list = []

    if isinstance(covar_module0, list):
        if type_KL == 'GPapprox':
            for sample in range(0, num_samples):
                Z = nnet_model.sample_latent(full_mu, full_log_var)
                for i in range(0, latent_dim):
                    Z_dim = Z[:, i]
                    gp_loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], full_labels, Z_dim,
                                    zt_list[i].to(device), P, T, eps)
                    gp_loss_sum = gp_loss.item() + gp_loss_sum
            gp_loss_sum /= num_samples

        elif type_KL == 'GPapprox_closed':
            for i in range(0, latent_dim):
                mu_sliced = full_mu[:, i]
                log_var_sliced = full_log_var[:, i]
                gp_loss = deviance_upper_bound(covar_module0[i], covar_module1[i],
                                               likelihoods[i], full_labels,
                                               mu_sliced, log_var_sliced,
                                               zt_list[i].to(device), P,
                                               T, eps)
                gp_loss_sum = gp_loss.item() + gp_loss_sum
    else:
        if type_KL == 'GPapprox_closed':
            gp_loss = validation_dubo(latent_dim, covar_module0, covar_module1,
                                      likelihoods, full_labels,
                                      full_mu, full_log_var,
                                      zt_list, P, T, eps)
            gp_loss_sum = gp_loss.item()

    if loss_function == 'mse':
        gp_loss_sum /= latent_dim
        net_loss_sum = weight*gp_loss_sum + recon_loss_sum
    elif loss_function == 'nll':
        net_loss_sum = gp_loss_sum + nll_loss_sum

    #Do logging
    print('Validation set - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
        net_loss_sum, gp_loss_sum, nll_loss_sum, recon_loss_sum))

    halfway = P//2
    l1 = [i*T + k for i in range(0,50) for k in range(0,5)]
    l2 = [i*T + k for i in range(halfway,halfway+50) for k in range(0,5)]
    prediction_mu = torch.cat((train_mu,
                               full_mu[l1],
                               full_mu[l2]))
    prediction_x = torch.cat((train_x,
                              full_labels[l1],
                              full_labels[l2]))
    test_x = torch.cat((full_labels[0:50*T], full_labels[halfway*T:(halfway+50)*T]))

    prediction_dataloader_p1 = DataLoader(dataset[0:50*T], batch_size=50*T, shuffle=False)
    for batch_idx, sample_batched in enumerate(prediction_dataloader_p1):
        data_p1 = sample_batched['digit'].double().to(device)
        mask_p1 = sample_batched['mask'].double().to(device)
    prediction_dataloader_p2 = DataLoader(dataset[halfway*T:(halfway+50)*T], batch_size=50*T, shuffle=False)
    for batch_idx, sample_batched in enumerate(prediction_dataloader_p2):
        data_p2 = sample_batched['digit'].double().to(device)
        mask_p2 = sample_batched['mask'].double().to(device)
    prediction_data = torch.cat((data_p1, data_p2))
    prediction_mask = torch.cat((mask_p1, mask_p2))

    Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
    recon_batch = nnet_model.decode(Z_pred)
    [recon_loss, nll] = nnet_model.loss_function(recon_batch, prediction_data, prediction_mask)
    print('Validation set - GP prediction MSE: %.3f' % (torch.sum(recon_loss).item()))
