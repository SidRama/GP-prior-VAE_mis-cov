from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import torch
import os

from elbo_functions import deviance_upper_bound, elbo, KL_closed, minibatch_KLD_upper_bound, \
    minibatch_KLD_upper_bound_iter, minibatch_sgd
from model_test import MSE_test_simple_batch
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from predict_HealthMNIST import recon_complete_gen, gen_rotated_mnist_plot, variational_complete_gen
from validation import validate, validate_simple_batch


def hensman_training(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_module0,
                     covar_module1, likelihoods, m, H, zt_list, P, T, varying_T, Q, weight, id_covariate, loss_function,
                     natural_gradient=False, natural_gradient_lr=0.01, subjects_per_batch=20, memory_dbg=False,
                     eps=1e-6, results_path=None, validation_dataset=None, generation_dataset=None, prediction_dataset=None,
                     labels_train=None):
    """
    Perform training with minibatching and Stochastic Variational Inference [Hensman et. al, 2013]. See L-VAE supplementary
    materials

    :param nnet_model: encoder/decoder neural network model
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param m: variational mean
    :param H: variational variance
    :param zt_list: list of inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param natural_gradient: use of natural gradients
    :param natural_gradient_lr: natural gradients learning rate
    :param subject_per_batch; number of subjects per batch (vectorisation)
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param results_path: path to results
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction
    :param gp_mode: GPyTorch gp model
    :param csv_file_test_data: path to test data
    :param csv_file_test_label: path to test label
    :param test_mask_file: path to test mask
    :param data_source_path: path to data source

    :return trained models and resulting losses

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)
    assert type_KL == 'GPapprox_closed'

    def f(x):
        return int(x['label'][id_covariate].item())
    P = len(set(list(map(f, dataset))))

    if varying_T:
        n_batches = (P + subjects_per_batch - 1)//subjects_per_batch
        dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=4)    
    else:
        batch_size = subjects_per_batch*T
        n_batches = (P*T + batch_size - 1)//(batch_size)
        dataloader = HensmanDataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P, T), batch_size, drop_last=False), num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        iid_kld_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader):
            optimiser.zero_grad()

            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]

            covariates = train_x[:, 2:6]
            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]

            train_label_means = torch.sum(covariates, dim=0) / torch.sum(covariates_mask, dim=0)
            covariates_mask_bool = torch.gt(covariates_mask, 0)
            train_label_std = torch.zeros(covariates.shape[1])
            for col in range(0, covariates.shape[1]):
                train_label_std[col] = torch.std(covariates[covariates_mask_bool[:, col], col])

            covariates_norm = (covariates - train_label_means) / train_label_std
            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            recon_batch, mu, log_var, X_tilde = nnet_model(data, covariates_norm)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates_norm * covariates_mask


            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))

            if varying_T:
                P_in_current_batch = torch.unique(train_x[:, id_covariate]).shape[0]
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound_iter(covar_module0, covar_module1, likelihoods, latent_dim, m, PSD_H, train_x, mu, log_var, zt_list, P, P_in_current_batch, N, natural_gradient, id_covariate, eps)
            else:
                P_in_current_batch = N_batch // T
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound(covar_module0, covar_module1, likelihoods, latent_dim, m, PSD_H, X_hat, mu, log_var, zt_list, P, P_in_current_batch, T, natural_gradient, eps)

            recon_loss = recon_loss * P/P_in_current_batch
            nll_loss = nll_loss * P/P_in_current_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr*(grad_H + grad_H.transpose(-1,-2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr*(grad_m - 2*torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches 
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr,  net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        if (not epoch % 20) and epoch != epochs:
            with torch.no_grad():
                if validation_dataset is not None:
                    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
                    prediction_x = torch.zeros(len(dataset), Q, dtype=torch.double).to(device)
                    for batch_idx, sample_batched in enumerate(dataloader):
                        label_id = sample_batched['idx']
                        prediction_x[label_id] = sample_batched['label'].double().to(device)
                        data = sample_batched['digit'].double().to(device)
                        covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                        if type_nnet == 'rnn':
                            mu, log_var = nnet_model.encode(data, covariates, varying_T=True, subjects=prediction_x[label_id, id_covariate])
                        else:
                            mu, log_var = nnet_model.encode(data)
                        full_mu[label_id] = mu
                    validate(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, weight, full_mu, prediction_x, id_covariate, loss_function, eps=1e-6)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H



def hensman_training_impute(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_module0,
                         covar_module1, likelihoods, m, H, zt_list, P, T, varying_T, Q, weight, id_covariate, loss_function,
                         natural_gradient=False, natural_gradient_lr=0.01, subjects_per_batch=20, memory_dbg=False, eps=1e-6,
                         file_num=1, csv_file_validation_data='validation_data.csv', csv_file_validation_label='validation_labels_unmasked.csv',
                         validation_mask_file='validation_mask.csv', data_source_path='./data',
                         csv_file_test_data='test_data.csv', csv_file_test_label='test_labels_unmasked.csv', test_mask_file='test_mask.csv',
                         results_path=None, validation_dataset=None, generation_dataset=None, prediction_dataset=None, labels_train=None):
    """
    Perform training with minibatching and Stochastic Variational Inference [Hensman et. al, 2013]. See L-VAE supplementary
    materials. This method also performs missing covariate imputation.

    :param nnet_model: encoder/decoder neural network model
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param m: variational mean
    :param H: variational variance
    :param zt_list: list of inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param natural_gradient: use of natural gradients
    :param natural_gradient_lr: natural gradients learning rate
    :param subject_per_batch; number of subjects per batch (vectorisation)
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param results_path: path to results
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction
    :param gp_mode: GPyTorch gp model
    :param csv_file_test_data: path to test data
    :param csv_file_test_label: path to test label
    :param test_mask_file: path to test mask
    :param data_source_path: path to data source

    :return trained models and resulting losses

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)

    batch_size = 50
    n_batches = (N + batch_size - 1) // (batch_size)
    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    if loss_function == 'mse':
        valid_best = np.Inf
    else:
        valid_best = np.Inf

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        iid_kld_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader):
            optimiser.zero_grad()

            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]

            covariates = train_x[:, 2:6]

            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]

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
            se = torch.mul(loss(X_tilde_norm.view(-1, covariates.shape[1]), covariates_norm.view(-1, covariates.shape[1])), covariates_mask.view(-1, covariates.shape[1]))
            mask_sum = torch.sum(covariates_mask.view(-1, covariates.shape[1]), dim=1)
            mask_sum[mask_sum == 0] = 1
            mse_X = torch.sum(torch.sum(se, dim=1) / mask_sum)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

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

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                            grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)
        train_label_means = torch.tensor([])
        train_label_std = torch.tensor([])

        if (not epoch % 20) and epoch != epochs:
            with torch.no_grad():
                if validation_dataset is not None:
                    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
                    prediction_x = torch.zeros(len(dataset), Q-2, dtype=torch.double).to(device)
                    for batch_idx, sample_batched in enumerate(dataloader):
                        label_id = sample_batched['idx']
                        data = sample_batched['digit'].double().to(device)
                        train_x = sample_batched['label'].double().to(device)
                        covariates = train_x[:, 2:6]

                        label_mask = sample_batched['label_mask'].double()
                        label_mask = label_mask.to(device)
                        covariates_mask = label_mask[:, 0:4]

                        train_label_means = torch.sum(covariates, dim=0) / torch.sum(covariates_mask, dim=0)
                        covariates_mask_bool = torch.gt(covariates_mask, 0)
                        train_label_std = torch.zeros(covariates.shape[1])
                        for col in range(0, covariates.shape[1]):
                            train_label_std[col] = torch.std(covariates[covariates_mask_bool[:, col], col])

                        covariates_norm = (covariates - train_label_means) / train_label_std
                        noise_replace = torch.zeros_like(covariates_norm)

                        covariates_norm = covariates_norm * covariates_mask
                        covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

                        recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm,
                                                                                            covariates_mask)
                        X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
                        X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates * covariates_mask

                        prediction_x[label_id] = X_hat
                        full_mu[label_id] = mu
                    validate_simple_batch(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim,
                                          covar_module0, likelihoods, zt_list, weight, full_mu, prediction_x, loss_function,
                                          batch_size, m, H, natural_gradient, eps=1e-6)
                    print("Running tests with a validation set")
                    # validation test
                    valid_mse, valid_nll = MSE_test_simple_batch(csv_file_validation_data, csv_file_validation_label, validation_mask_file,
                                                                 data_source_path, nnet_model, covar_module0, likelihoods,
                                                                 results_path, file_num, prediction_x, full_mu, zt_list, N,
                                                                 train_label_means, train_label_std, latent_dim,
                                                                 save_csv='result_error_valid.csv', save_img='recon_complete_valid.pdf')

                    if loss_function == 'mse' and valid_mse < valid_best:
                        print('Best validation score updated')
                        valid_best = valid_mse
                        print("Running tests with a test set")
                        MSE_test_simple_batch(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path,
                                              nnet_model,covar_module0, likelihoods, results_path, file_num, prediction_x, full_mu,
                                              zt_list, N, train_label_means, train_label_std, latent_dim, save_csv='result_error_best.csv',
                                              save_img='recon_complete_best.pdf')

                    elif loss_function == 'nll' and valid_nll < valid_best:
                        print('Best validation score updated')
                        print("Running tests with a test set")
                        valid_best = valid_nll
                        MSE_test_simple_batch(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path,
                                              nnet_model, covar_module0, likelihoods, results_path, file_num,
                                              prediction_x, full_mu,
                                              zt_list, N, train_label_means, train_label_std, latent_dim,
                                              save_csv='result_error_best.csv',
                                              save_img='recon_complete_best.pdf')



                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list



def minibatch_training(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_module0,
                       covar_module1, likelihoods, zt_list, P, T, Q, weight, id_covariate, loss_function, memory_dbg=False,
                       eps=1e-6, results_path=None, validation_dataset=None, generation_dataset=None, prediction_dataset=None):
    """
    Perform training with minibatching (psuedo-minibatching) similar to GPPVAE [Casale el. al, 2018]. See L-VAE supplementary
    materials

    :param nnet_model: encoder/decoder neural network model
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param zt_list: list of inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param results_path: path to results
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction

    :return trained models and resulting losses

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = T
    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    gp_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    for epoch in range(1, epochs + 1):

        optimiser.zero_grad()

        full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
        full_log_var = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
        train_x = torch.zeros(len(dataset), Q, dtype=torch.double, requires_grad=False).to(device)

        #Step 1: Encode the sample data to obtain \bar{\mu} and diag(W)
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(dataloader):
                indices = sample_batched['idx']
                data = sample_batched['digit'].double().to(device)
                train_x[indices] = sample_batched['label'].double().to(device)

                covariates = torch.cat((train_x[indices, :id_covariate], train_x[indices, id_covariate+1:]), dim=1)
                if type_nnet == 'rnn':
                    mu, log_var = nnet_model.encode(data, covariates)
                else:
                    mu, log_var = nnet_model.encode(data)

                full_mu[indices] = mu
                full_log_var[indices] = log_var

        mu_grads = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)
        log_var_grads = torch.zeros(len(dataset), latent_dim, dtype=torch.double, requires_grad=True).to(device)

        gp_losses = 0
        gp_loss_sum = 0
        param_list = []

        #Steps 2 & 3: compute d and E, compute gradients of KLD w.r.t S and theta
        if type_KL == 'GPapprox':
            for sample in range(0, num_samples):
                Z = nnet_model.sample_latent(full_mu, full_log_var)
                for i in range(0, latent_dim):
                    Z_dim = Z[:, i]
                    gp_loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], train_x, Z_dim,
                                    zt_list[i].to(device), P, T, eps)
                    gp_loss_sum = gp_loss.item() + gp_loss_sum
                    gp_losses = gp_losses + gp_loss
            gp_losses = gp_losses / num_samples
            gp_loss_sum /= num_samples

        elif type_KL == 'GPapprox_closed':
            for i in range(0, latent_dim):
                mu_sliced = full_mu[:, i]
                log_var_sliced = full_log_var[:, i]
                gp_loss = deviance_upper_bound(covar_module0[i], covar_module1[i],
                                               likelihoods[i], train_x,
                                               mu_sliced, log_var_sliced,
                                               zt_list[i].to(device), P,
                                               T, eps)
                gp_loss_sum = gp_loss.item() + gp_loss_sum
                gp_losses = gp_losses + gp_loss

        
        for i in range(0, latent_dim):
            param_list += list(covar_module0[i].parameters())
            param_list += list(covar_module1[i].parameters())

        if loss_function == 'mse':
            gp_losses = weight*gp_losses/latent_dim
            gp_loss_sum /= latent_dim
        
        mu_grads = torch.autograd.grad(gp_losses, full_mu, retain_graph=True)[0]
        log_var_grads = torch.autograd.grad(gp_losses, full_log_var, retain_graph=True)[0]
        grads = torch.autograd.grad(gp_losses, param_list)

        for ind, p in enumerate(param_list):
            p.grad = grads[ind]

        recon_loss_sum = 0
        nll_loss_sum = 0
        #Step 4: compute reconstruction losses w.r.t phi and psi, add dKLD/dphi to the gradients
        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['digit'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            indices = sample_batched['idx']

            label = sample_batched['label'].double().to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)

            if type_nnet == 'rnn':
                recon_batch, mu, log_var = nnet_model(data, covariates)
            else:
                recon_batch, mu, log_var = nnet_model(data)
            
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll = torch.sum(nll)

            mu.backward(mu_grads[indices], retain_graph = True)
            log_var.backward(log_var_grads[indices], retain_graph = True)

            if loss_function == 'mse':         
                recon_loss.backward()
            elif loss_function == 'nll':
                nll.backward()
 
            recon_loss_sum = recon_loss_sum + recon_loss.item()
            nll_loss_sum = nll_loss_sum + nll.item()

        #Do logging
        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, recon_loss_sum + weight*gp_loss_sum, gp_loss_sum, nll_loss_sum, recon_loss_sum))
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr,  recon_loss_sum + weight*gp_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        gp_loss_arr = np.append(gp_loss_arr, gp_loss_sum)

        #Step 5: apply gradients using an Adam optimiser
        optimiser.step()

        if (not epoch % 10) and epoch != epochs:
            if validation_dataset is not None:
                validate(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, weight, full_mu, train_x, id_covariate, loss_function, eps=1e-6)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if results_path and generation_dataset:
                prediction_dataloader = DataLoader(prediction_dataset, batch_size=1000, shuffle=False, num_workers=4)
                full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
                prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)
                with torch.no_grad():
                    for batch_idx, sample_batched in enumerate(prediction_dataloader):
                        # no mini-batching. Instead get a batch of dataset size
                        label_id = sample_batched['idx']
                        prediction_x[label_id] = sample_batched['label'].double().to(device)
                        data = sample_batched['digit'].double().to(device)
                        covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                        if type_nnet == 'rnn':
                            mu, log_var = nnet_model.encode(data, covariates, varying_T=True, subjects=prediction_x[label_id, id_covariate])
                        else:
                            mu, log_var = nnet_model.encode(data)

                        full_mu[label_id] = mu

                    recon_complete_gen(generation_dataset, nnet_model, type_nnet,
                                       results_path, covar_module0,
                                       covar_module1, likelihoods, latent_dim,
                                       './data', prediction_x, full_mu, epoch,
                                       zt_list, P, T, id_covariate)

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr

def standard_training(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_modules,
                      likelihoods, zt_list, id_covariate, P, T, Q, weight, constrain_scales, loss_function, memory_dbg=False,
                      eps=1e-6, validation_dataset=None, generation_dataset=None, prediction_dataset=None, time_age_train=None):
    """
    Perform training without minibatching.

    :param nnet_model: encoder/decoder neural network model
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim: number of latent dimensions
    :param covar_modules: additive kernel (sum of cross-covariances)
    :param likelihoods: GPyTorch likelihood model
    :param zt_list: list of inducing points
    :param id_covariate: covariate number of the id
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param constrain_scales: boolean to constrain scales to 1
    :param loss_function: selected loss function
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction

    :return trained models and resulting losses

    """
    if type_KL == 'closed':
        covar_module = covar_modules[0]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_module0 = covar_modules[0]
        covar_module1 = covar_modules[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    gp_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    for epoch in range(1, epochs + 1):
        for batch_idx, sample_batched in enumerate(dataloader):

            # no mini-batching. Instead get a batch of dataset size.
            optimiser.zero_grad()                                       # clear gradients
            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            train_x = label.double().to(device)

            covariates = train_x[:, 2:6]

            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            #            label_mask[:, 3] = torch.ones(200)
            covariates_mask = label_mask[:, 0:4]

            #            cov_nnet_model.eval()

            covariates_norm = (covariates - train_label_means) / train_label_std
            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            # encode data

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            gp_loss_avg = torch.tensor([0.0]).to(device)
            net_loss = torch.tensor([0.0]).to(device)
            penalty_term = torch.tensor([0.0]).to(device)

            for sample_iter in range(0, num_samples):

                # Iterate over specified number of samples. Default: num_samples = 1.
                Z = nnet_model.sample_latent(mu, log_var)
                gp_loss = torch.tensor([0.0]).to(device)

                for i in range(0, latent_dim):
                    Z_dim = Z[:, i].view(-1).type(torch.DoubleTensor).to(device)

                    if type_KL == 'closed':

                        # Closed-form KL divergence formula
                        kld1 = KL_closed(covar_module[i], train_x, likelihoods[i], data,  mu[:, i], log_var[:, i])
                        gp_loss = gp_loss + kld1
                    elif type_KL == 'conj_gradient':

                        # GPyTorch default: use modified batch conjugate gradients
                        # See: https://arxiv.org/abs/1809.11165
                        gp_models[i].set_train_data(train_x.to(device), Z_dim.to(device))
                        gp_loss = gp_loss - mlls[i](gp_models[i](train_x.to(device)), Z_dim)
                    elif type_KL == 'GPapprox':

                        # Our proposed efficient approximate GP inference scheme
                        # See: http://arxiv.org/abs/2006.09763
                        loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], train_x, Z_dim,
                                     zt_list[i].to(device), P, T, eps)
                        gp_loss = gp_loss + loss

                    elif type_KL == 'GPapprox_closed':

                        # A variant of our proposed efficient approximate GP inference scheme.
                        # The key difference with GPapprox is the direct use of the variational mean and variance,
                        # instead of a sample from Z. We can call this a deviance upper bound.
                        # See the L-VAE supplement for more details: http://arxiv.org/abs/2006.09763
                        loss = deviance_upper_bound(covar_module0[i], covar_module1[i], likelihoods[i], train_x,
                                                    mu[:, i].view(-1), log_var[:, i].view(-1), zt_list[i].to(device), P,
                                                    T, eps)
                        gp_loss = gp_loss + loss


                if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                    if loss_function == 'mse':
                        gp_loss_avg = gp_loss_avg + (gp_loss / latent_dim)
                    elif loss_function == 'nll':
                        gp_loss_avg = gp_loss_avg + gp_loss
                elif type_KL == 'conj_gradient':
                    if loss_function == 'mse':
                        gp_loss = gp_loss * data.shape[0] / latent_dim
                    elif loss_function == 'nll':
                        gp_loss = gp_loss * data.shape[0]
                    gp_loss_avg = gp_loss_avg + gp_loss

            if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                gp_loss_avg = gp_loss_avg / num_samples
                if loss_function == 'mse':
                    net_loss = recon_loss + weight * gp_loss_avg
                elif loss_function == 'nll':
                    net_loss = nll_loss + gp_loss_avg
            elif type_KL == 'conj_gradient':
                gp_loss_avg = gp_loss_avg / num_samples
                penalty_term = -0.5 * log_var.sum() / latent_dim
                if loss_function == 'mse':
                    net_loss = recon_loss + weight * (gp_loss_avg + penalty_term)
                elif loss_function == 'nll':
                    net_loss = nll_loss + gp_loss_avg + penalty_term

            net_loss.backward()

            if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
                    epoch, epochs, net_loss.item(), gp_loss_avg.item(), nll_loss.item(), recon_loss.item()))
            elif type_KL == 'conj_gradient':
                print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - Penalty: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
                    epoch, epochs, net_loss.item(), gp_loss_avg.item(), penalty_term.item(), nll_loss.item(), recon_loss.item()))

            penalty_term_arr = np.append(penalty_term_arr, penalty_term.cpu().item())
            net_train_loss_arr = np.append(net_train_loss_arr, net_loss.cpu().item())
            recon_loss_arr = np.append(recon_loss_arr, recon_loss.cpu().item())
            nll_loss_arr = np.append(nll_loss_arr, nll_loss.cpu().item())
            gp_loss_arr = np.append(gp_loss_arr, gp_loss_avg.cpu().item())
            optimiser.step()
            if constrain_scales:
                for i in range(0, latent_dim):
                    likelihoods[i].noise = torch.tensor([1], dtype=torch.float).to(device)

            if (not epoch % 100) and epoch != epochs:
                if validation_dataset is not None:
                    standard_validate(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, weight, mu, train_x, id_covariate, loss_function, eps=1e-6)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr


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


def standard_training_impute(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_modules,
                      likelihoods, zt_list, id_covariate, P, T, Q, weight, constrain_scales, loss_function, memory_dbg=False,
                      eps=1e-6, validation_dataset=None, generation_dataset=None, prediction_dataset=None,
                      train_label_means=None, train_label_std=None, time_age_train=None, labels_train=None):
    """
     Perform training without minibatching. This method performs imputation of missing covariates

     :param nnet_model: encoder/decoder neural network model
     :param type_nnet: type of encoder/decoder
     :param epochs: numner of epochs
     :param dataset: dataset to use in training
     :param optimiser: optimiser to be used
     :param type_KL: type of KL divergenve computation to use
     :param num_samples: number of samples to use
     :param latent_dim: number of latent dimensions
     :param covar_modules: additive kernel (sum of cross-covariances)
     :param likelihoods: GPyTorch likelihood model
     :param zt_list: list of inducing points
     :param id_covariate: covariate number of the id
     :param P: number of unique instances
     :param T: number of longitudinal samples per individual
     :param Q: number of covariates
     :param weight: value for the weight
     :param constrain_scales: boolean to constrain scales to 1
     :param loss_function: selected loss function
     :param memory_dbg: enable debugging
     :param eps: jitter
     :param validation_dataset: dataset for vaildation set
     :param generation_dataset: dataset to help with sample image generation
     :param prediction_dataset; dataset with subjects for prediction

     :return trained models and resulting losses

     """
    weight = 0.001
    if type_KL == 'closed':
        covar_module = covar_modules[0]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_module0 = covar_modules[0]
        covar_module1 = covar_modules[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    gp_loss_arr = np.empty((0, 1))
    mse_label_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    for epoch in range(1, epochs + 1):
        for batch_idx, sample_batched in enumerate(dataloader):

            # no mini-batching. Instead get a batch of dataset size.
            optimiser.zero_grad()                                       # clear gradients
            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            train_x = label.double().to(device)

            covariates = train_x[:, 2:6]

            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]


            covariates_norm = (covariates - train_label_means) / train_label_std
            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, covariates_mask)
            X_tilde_norm = X_tilde * (1 - covariates_mask) + covariates_norm * covariates_mask
            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates * covariates_mask

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

            for sample_iter in range(0, num_samples):

                # Iterate over specified number of samples. Default: num_samples = 1.
                Z = nnet_model.sample_latent(mu, log_var)
                gp_loss = torch.tensor([0.0]).to(device)

                for i in range(0, latent_dim):
                    Z_dim = Z[:, i].view(-1).type(torch.DoubleTensor).to(device)

                    if type_KL == 'closed':

                        # Closed-form KL divergence formula
                        kld1 = KL_closed(covar_module[i], X_hat, likelihoods[i], data,  mu[:, i], log_var[:, i])
                        gp_loss = gp_loss + kld1
                    elif type_KL == 'conj_gradient':

                        # GPyTorch default: use modified batch conjugate gradients
                        # See: https://arxiv.org/abs/1809.11165
                        gp_models[i].set_train_data(train_x.to(device), Z_dim.to(device))
                        gp_loss = gp_loss - mlls[i](gp_models[i](train_x.to(device)), Z_dim)
                    elif type_KL == 'GPapprox':

                        # Our proposed efficient approximate GP inference scheme
                        # See: http://arxiv.org/abs/2006.09763
                        loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], train_x, Z_dim,
                                     zt_list[i].to(device), P, T, eps)
                        gp_loss = gp_loss + loss

                    elif type_KL == 'GPapprox_closed':

                        # A variant of our proposed efficient approximate GP inference scheme.
                        # The key difference with GPapprox is the direct use of the variational mean and variance,
                        # instead of a sample from Z. We can call this a deviance upper bound.
                        # See the L-VAE supplement for more details: http://arxiv.org/abs/2006.09763
                        loss = deviance_upper_bound(covar_module0[i], covar_module1[i], likelihoods[i], train_x,
                                                    mu[:, i].view(-1), log_var[:, i].view(-1), zt_list[i].to(device), P,
                                                    T, eps)
                        gp_loss = gp_loss + loss
                gp_loss = gp_loss + kl_X

                if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                    if loss_function == 'mse':
                        gp_loss_avg = gp_loss_avg + (gp_loss / latent_dim)
                    elif loss_function == 'nll':
                        gp_loss_avg = gp_loss_avg + gp_loss
                elif type_KL == 'conj_gradient':
                    if loss_function == 'mse':
                        gp_loss = gp_loss * data.shape[0] / latent_dim
                    elif loss_function == 'nll':
                        gp_loss = gp_loss * data.shape[0]
                    gp_loss_avg = gp_loss_avg + gp_loss

            if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                gp_loss_avg = gp_loss_avg / num_samples
                if loss_function == 'mse':
                    net_loss = recon_loss + weight * gp_loss_avg
                elif loss_function == 'nll':
                    net_loss = nll_loss + gp_loss_avg
            elif type_KL == 'conj_gradient':
                gp_loss_avg = gp_loss_avg / num_samples
                penalty_term = -0.5 * log_var.sum() / latent_dim
                if loss_function == 'mse':
                    net_loss = recon_loss + weight * (gp_loss_avg + penalty_term)
                elif loss_function == 'nll':
                    net_loss = nll_loss + gp_loss_avg + penalty_term

            loss = nn.MSELoss(reduction='none')
            se = torch.mul(loss(X_hat, labels_train),
                           torch.ones_like(label_mask, dtype=torch.double))
            mask_sum = torch.sum(label_mask, dim=1)
            mse_label_loss = torch.mean(torch.sum(se, dim=1) / mask_sum)

            net_loss.backward()

            if type_KL == 'closed' or type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f - Label loss: %.3f - NLL Loss: %.3f  - Recon Loss: %.3f' % (
                    epoch, epochs, net_loss.item(), gp_loss_avg.item(), mse_label_loss.item(), nll_loss.item(), recon_loss.item()))
            elif type_KL == 'conj_gradient':
                print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - Penalty: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
                    epoch, epochs, net_loss.item(), gp_loss_avg.item(), penalty_term.item(), nll_loss.item(), recon_loss.item()))

            penalty_term_arr = np.append(penalty_term_arr, penalty_term.cpu().item())
            net_train_loss_arr = np.append(net_train_loss_arr, net_loss.cpu().item())
            recon_loss_arr = np.append(recon_loss_arr, recon_loss.cpu().item())
            nll_loss_arr = np.append(nll_loss_arr, nll_loss.cpu().item())
            gp_loss_arr = np.append(gp_loss_arr, gp_loss_avg.cpu().item())
            mse_label_loss_arr = np.append(mse_label_loss_arr, mse_label_loss.cpu().item())
            optimiser.step()
            if constrain_scales:
                for i in range(0, latent_dim):
                    likelihoods[i].noise = torch.tensor([1], dtype=torch.float).to(device)

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr, mse_label_loss_arr

def variational_inference_optimization(nnet_model, type_nnet, epochs, dataset, prediction_dataset, optimiser, latent_dim,
                                       covar_module0, covar_module1, likelihoods, zt_list, P, T, Q, weight, constrain_scales,
                                       id_covariate, loss_function, memory_dbg=False, eps=1e-6, results_path=None,
                                       save_path=None, gp_model_folder=None, generation_dataset=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    gp_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))

    for batch_idx, sample_batched in enumerate(dataloader):
        label_id = sample_batched['idx']
        label = sample_batched['label'].double().to(device)
        data = sample_batched['digit'].double().to(device)
        mask = sample_batched['mask'].double().to(device)

        covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)

        # encode data
        if type_nnet == 'rnn':
            mu, log_var = nnet_model.encode(data, covariates)
        else:
            mu, log_var = nnet_model.encode(data)

    mu = torch.nn.Parameter(mu.clone().detach(), requires_grad=True)
    log_var = torch.nn.Parameter(log_var.clone().detach(), requires_grad=True)

    try:
        mu = torch.load(os.path.join(gp_model_folder, 'mu.pth'), map_location=torch.device(device)).detach().to(device).requires_grad_(True)
        log_var = torch.load(os.path.join(gp_model_foder, 'log_var.pth'), map_location=torch.device(device)).detach().to(device).requires_grad_(True)
    except:
        pass

    optimiser.add_param_group({'params': mu})
    optimiser.add_param_group({'params': log_var})

    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        Z = nnet_model.sample_latent(mu, log_var)
        recon_batch = nnet_model.decode(Z)
        [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
        recon_loss = torch.sum(recon_loss)
        nll_loss = torch.sum(nll)

        gp_loss_avg = torch.tensor([0.0]).to(device)
        net_loss = torch.tensor([0.0]).to(device)
        penalty_term = torch.tensor([0.0]).to(device)

        for i in range(0, latent_dim):
            loss = deviance_upper_bound(covar_module0[i], covar_module1[i], likelihoods[i], label,
                                        mu[:, i].view(-1), log_var[:, i].view(-1), zt_list[i].to(device), P,
                                        T, eps)
            gp_loss_avg = gp_loss_avg + loss / latent_dim

        if loss_function == 'mse':
            net_loss = recon_loss + weight * gp_loss_avg
        elif loss_function == 'nll':
            net_loss = nll_loss + gp_loss_avg

        net_loss.backward()

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
              epoch, epochs, net_loss.item(), gp_loss_avg.item(), nll_loss.item(), recon_loss.item()),
              flush=True)

        penalty_term_arr = np.append(penalty_term_arr, penalty_term.cpu().item())
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss.cpu().item())
        recon_loss_arr = np.append(recon_loss_arr, recon_loss.cpu().item())
        nll_loss_arr = np.append(nll_loss_arr, nll_loss.cpu().item())
        gp_loss_arr = np.append(gp_loss_arr, gp_loss_avg.cpu().item())
        optimiser.step()

        if not epoch % 100:
            sv_pth = os.path.join(save_path, 'recon_' + str(epoch) + '.pdf')
            gen_rotated_mnist_plot(data[1920:2080].cpu().detach(), recon_batch[1920:2080].cpu().detach(), label[1920:2080].cpu().detach(), seq_length=20, num_sets=8, save_file=sv_pth)

    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'final-vae_model.pth'))
    torch.save(mu, os.path.join(save_path, 'mu.pth'))
    torch.save(log_var, os.path.join(save_path, 'log_var.pth'))
    for i in range(0, latent_dim):
        torch.save(covar_module0[i].state_dict(), os.path.join(save_path, 'cov_module0_' + str(i) + '.pth'))
        torch.save(covar_module1[i].state_dict(), os.path.join(save_path, 'cov_module1_' + str(i) + '.pth'))

    prediction_dataloader = DataLoader(prediction_dataset, batch_size=len(prediction_dataset), shuffle=False, num_workers=1)
    for batch_idx, sample_batched in enumerate(prediction_dataloader):
        label_pred = sample_batched['label'].double().to(device)
        data_pred = sample_batched['digit'].double().to(device)
        mask_pred = sample_batched['mask'].double().to(device)
        covariates = torch.cat((label_pred[:, :id_covariate], label_pred[:, id_covariate+1:]), dim=1)
        # encode data
        if type_nnet == 'rnn':
            mu_pred, log_var_pred = nnet_model.encode(data_pred, covariates)
        else:
            mu_pred, log_var_pred = nnet_model.encode(data_pred)
        break

    try:
        mu_pred = torch.load(os.path.join(gp_model_folder, 'mu_pred.pth'), map_location=torch.device(device)).detach().to(device).requires_grad_(True)
        log_var_pred = torch.load(os.path.join(gp_model_folder, 'log_var_pred.pth'), map_location=torch.device(device)).detach().to(device).requires_grad_(True)
    except:
        pass

    mu_pred = torch.nn.Parameter(mu_pred.clone().detach(), requires_grad=True)
    log_var_pred = torch.nn.Parameter(log_var_pred.clone().detach(), requires_grad=True)
    adam_param_list = []
    adam_param_list.append({'params': mu_pred})
    adam_param_list.append({'params': log_var_pred})
    optimiser_pred = torch.optim.Adam(adam_param_list, lr=1e-3)
    for epoch in range(1, 1001):
        optimiser_pred.zero_grad()

        Z = nnet_model.sample_latent(mu_pred, log_var_pred)

        recon_batch = nnet_model.decode(Z)
        [recon_loss, nll] = nnet_model.loss_function(recon_batch,
                                                     data_pred,
                                                     mask_pred)

        recon_loss = torch.sum(recon_loss)
        nll_loss = torch.sum(nll)

        gp_loss_avg = torch.tensor([0.0]).to(device)

        prediction_mu = torch.cat((mu_pred, mu), dim=0)
        prediction_log_var = torch.cat((log_var_pred, log_var), dim=0)
        prediction_x = torch.cat((label_pred, label), dim=0)

        for i in range(0, latent_dim):
            loss = deviance_upper_bound(covar_module0[i], covar_module1[i], likelihoods[i], prediction_x,
                                        prediction_mu[:, i].view(-1), prediction_log_var[:, i].view(-1),
                                        zt_list[i].to(device), P+8, T, eps)
            gp_loss_avg = gp_loss_avg + loss / latent_dim

        if loss_function == 'mse':
            net_loss = recon_loss + weight * gp_loss_avg
        elif loss_function == 'nll':
            net_loss = nll_loss + gp_loss_avg

        net_loss.backward()

        print('Iter %d/1000 - Total Loss: %.3f  - GP Loss: %.3f  - Recon Loss: %.3f' % (
              epoch, net_loss.item(), gp_loss_avg.item(), recon_loss.item()),
              flush=True)

        optimiser_pred.step()

    torch.save(mu_pred, os.path.join(save_path, 'mu_pred.pth'))
    torch.save(log_var_pred, os.path.join(save_path, 'log_var_pred.pth'))

    l = [i*20 + k for i in range(0,8) for k in range(0,5)]
    prediction_x = torch.cat((label_pred[l],
                               label))
    prediction_mu = torch.cat((mu_pred[l],
                               mu))

    if generation_dataset:
        variational_complete_gen(generation_dataset, nnet_model, type_nnet,
                                 results_path, covar_module0,
                                 covar_module1, likelihoods, latent_dim,
                                 './data', prediction_x, prediction_mu, 'final',
                                 zt_list, P, T, id_covariate)

    exit(0)
