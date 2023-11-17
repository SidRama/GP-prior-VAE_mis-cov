import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils import batch_predict_varying_T


def gen_rotated_mnist_plot(X, recon_X, labels, seq_length=16, num_sets=3, save_file='recon.pdf'):
    """
    Function to generate rotated MNIST digits plots.

    """
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    fig.set_size_inches(9, 1.5 * num_sets)
    for j in range(num_sets):
        begin = seq_length * j
        end = seq_length * (j + 1)
        time_steps = labels[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')

def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate rotated MNIST digits.

    """
    num_sets = 8
    fig, ax = plt.subplots(4 * num_sets - 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)

    for j in range(num_sets):
        begin_data = seq_length_train*j
        end_data = seq_length_train*(j+1)

        begin_label = seq_length_full*2*j
        mid_label = seq_length_full*(2*j+1)
        end_label = seq_length_full*2*(j+1)
        
        time_steps = labels_train[begin_data:end_data, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j, int(t)].imshow(np.reshape(X[begin_data + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[begin_label:mid_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 1, int(t)].imshow(np.reshape(recon_X[begin_label + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[mid_label:end_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 2, int(t)].imshow(np.reshape(recon_X[mid_label + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')

def recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1, likelihoods,
                       latent_dim, data_source_path, prediction_x, prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
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
            #                train_label_std[col] = torch.std(covariates[:, col])

            covariates_norm = (covariates - train_label_means) / train_label_std

            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            if type_nnet == 'rnn':
                recon_batch, mu, log_var = nnet_model(data, covariates, varying_T=True, subjects=label[:, id_covariate])
            else:
                recon_batch, mu, log_var, X_tilde = nnet_model(data, covariates_norm)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates_norm * covariates_mask
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x,
                                             X_hat, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)

            filename = 'recon_complete.pdf' if epoch == -1 else 'recon_complete_' + str(epoch) + '.pdf'

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, filename))

def recon_complete_simple(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1,
                          likelihoods, latent_dim, data_source_path, prediction_x, prediction_mu, epoch, zt_list, P, T,
                          id_covariate, varying_T=False, file_num=None):
    """
    Function to generate rotated MNIST digits.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)

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

            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            if type_nnet == 'rnn':
                recon_batch, mu, log_var = nnet_model(data, covariates, varying_T=True, subjects=label[:, id_covariate])
            else:
                recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm)

            filename = 'recon_complete.pdf' if epoch == -1 else 'recon_complete_' + str(epoch) + '.pdf'
            filename = os.path.join(results_path, file_num, filename)

            fig, ax = plt.subplots(10, 20, figsize=(12, 12))
            for i in range(0, 200):
                z1_plot = ax[int(i / 20)][i % 20].imshow(np.reshape(recon_batch[i, :], [36, 36]), cmap='gray')
                #   print(labels[i, :])
                ax[int(i / 20)][i % 20].axis('off')

            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')


def variational_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1,
                             likelihoods, latent_dim, data_source_path, prediction_x, prediction_mu, epoch, zt_list, P, T,
                             id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)
            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete_' + str(epoch) + '.pdf'))

def VAEoutput(nnet_model, dataset, epoch, save_path, type_nnet, id_covariate):
    """
    Function to obtain output of VAE.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Batch size must be a multiple of T
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            # no mini-batching. Instead get a mini-batch of size 4000
            label = sample_batched['label'].to(device)
            data = sample_batched['digit'].to(device)

            recon_batch, mu, log_var = nnet_model(data)

            gen_rotated_mnist_plot(data[40:200, :].cpu(), recon_batch[40:200, :].cpu(), label[40:200, :].cpu(), seq_length=20, num_sets=8,
                                   save_file=os.path.join(save_path, 'recon_VAE_' + str(epoch) + '.pdf'))
            break
