import os
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset_def import HMNIST_dataset
from utils import  batch_predict_varying_T, predict_scalable


def predict_gp(kernel_component, full_kernel_inverse, z):
    """
    Function to compute predictive mean

    """

    mean = torch.matmul(torch.matmul(full_kernel_inverse, kernel_component), z)
    return mean


def MSE_test_simple(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                    covar_module, likelihoods, results_path, latent_dim, dataset_type, file_num, train_x, train_z):
    """
    Function to compute Mean Squared Error of test set.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'SimpleMNIST':
            test_dataset = HMNIST_dataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']

            data = sample_batched['digit']
            data = data.double().to(device)

            mask = torch.ones_like(data, dtype=torch.double)
            mask = mask.to(device)

            test_x = sample_batched['label'].double().to(device)

            covariates_test = test_x[:, 2:6]

            label_mask = torch.ones_like(test_x, dtype=torch.double)
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]
            Z_pred = torch.tensor([], dtype=torch.double).to(device)

            for i in range(0, latent_dim):
                covar_module[i].eval()
                likelihoods[i].eval()
                K1 = covar_module[i](covariates_test.to(device), covariates_test.to(device)).evaluate() \
                                     + likelihoods[i].noise * torch.eye(covariates_test.shape[0]).to(device)
                LK1 = torch.cholesky(K1)
                iK1 = torch.cholesky_solve(torch.eye(covariates_test.shape[0], dtype=torch.double).to(device), LK1).to(device)
                kernel_component = covar_module[i](covariates_test.to(device), train_x.to(device)).evaluate()
                pred_means = predict_gp(kernel_component, iK1, train_z[:, i])
                Z_pred = torch.cat((Z_pred, pred_means.view(-1, 1)), 1)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (nll): ' + str(torch.mean(nll)))
            print('Decoder loss: ' + str(torch.mean(recon_loss)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(nll).cpu().numpy()])
            np.savetxt(os.path.join(results_path, file_num, 'result_error.csv'), pred_results)

            filename = 'recon_complete_test.pdf'

            filename = os.path.join(results_path, file_num, filename)

            fig, ax = plt.subplots(5, 10, figsize=(12, 12))
            for i in range(0, 50):
                z1_plot = ax[int(i / 10)][i % 10].imshow(np.reshape(recon_Z[i, :], [36, 36]), cmap='gray')
                #   print(labels[i, :])
                ax[int(i / 10)][i % 10].axis('off')

            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')


def MSE_test_simple_newKL(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                    covar_module, likelihoods, results_path, latent_dim, dataset_type, file_num, train_x, train_z, train_label_means, train_label_std):
    """
       Function to compute Mean Squared Error of test set with GP approximationö

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'SimpleMNIST':
            test_dataset = HMNIST_dataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']

            data = sample_batched['digit']
            data = data.double().to(device)

            data_missing = torch.zeros_like(data, dtype=torch.double).to(device)

            mask = torch.ones_like(data, dtype=torch.double)
            mask = mask.to(device)

            test_x = sample_batched['label'].double().to(device)
            test_x = test_x[:, 2:6]

            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]

            covariates_test = test_x * covariates_mask

            covariates_norm = (covariates_test - train_label_means) / train_label_std
            noise_replace = torch.randn_like(covariates_norm)

            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data_missing, covariates_norm, covariates_mask)
            X_tilde_norm = X_tilde * (1 - covariates_mask) + covariates_norm * covariates_mask
            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates_test * covariates_mask
            covariates_test = X_hat
            covariates_mask = torch.ones_like(test_x)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)

            for i in range(0, latent_dim):
                covar_module[i].eval()
                likelihoods[i].eval()
                K1 = covar_module[i](covariates_test.to(device), covariates_test.to(device)).evaluate() \
                                     + likelihoods[i].noise * torch.eye(covariates_test.shape[0]).to(device)
                LK1 = torch.cholesky(K1)
                iK1 = torch.cholesky_solve(torch.eye(covariates_test.shape[0], dtype=torch.double).to(device), LK1).to(device)
                kernel_component = covar_module[i](covariates_test.to(device), train_x.to(device)).evaluate()
                pred_means = predict_gp(kernel_component, iK1, train_z[:, i])
                Z_pred = torch.cat((Z_pred, pred_means.view(-1, 1)), 1)

            recon_Z = nnet_model.decode(Z_pred, X_tilde_norm)
            [recon_loss, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (nll): ' + str(torch.mean(nll)))
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            loss = nn.MSELoss(reduction='none')

            se = torch.mul(loss(covariates_test, test_x), 1 - label_mask)
            mask_sum = torch.sum(1 - label_mask, dim=1)
            mask_sum[mask_sum == 0] = 1
            mse_label = torch.sum(se, dim=1) / mask_sum

            print('Label error: ' + str(torch.mean(mse_label)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(nll).cpu().numpy(), torch.mean(mse_label).cpu().numpy()])
            np.savetxt(os.path.join(results_path, file_num, 'result_error.csv'), pred_results)

            filename = 'recon_complete_test.pdf'

            filename = os.path.join(results_path, file_num, filename)

            fig, ax = plt.subplots(5, 10, figsize=(12, 12))
            for i in range(0, 50):
                z1_plot = ax[int(i / 10)][i % 10].imshow(np.reshape(recon_Z[i, :], [36, 36]), cmap='gray')
                ax[int(i / 10)][i % 10].axis('off')

            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')


def MSE_test(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
             covar_module, likelihoods, results_path, latent_dim, prediction_x, prediction_mu, dataset_type, file_num):
    """
    Function to compute Mean Squared Error of test set.

    """
    print("Running tests with a test set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'SimpleMNIST':
            test_dataset = HMNIST_dataset(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor())


    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
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

            data_masked = data * mask.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

            recon_batch, mu, log_var, mu_x, log_var_X, X_tilde = nnet_model(data, covariates_norm)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, 1 - mask)  # reconstruction loss
            print('Decoder loss (nll): ' + str(torch.mean(nll)))
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(nll).cpu().numpy()])
            np.savetxt(os.path.join(results_path, file_num, 'result_error.csv'), pred_results)


def MSE_test_simple_batch(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, nnet_model,
                            covar_module, likelihoods, results_path, file_num, train_x, pred_mu, zt_list, N, train_label_means, train_label_std, latent_dim,
                          save_csv='result_error.csv', save_img='recon_complete_test.pdf'):

    """
    Function to compute Mean Squared Error of test set.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HMNIST_dataset(csv_file_data=csv_file_test_data,
                                  csv_file_label=csv_file_test_label,
                                  mask_file=test_mask_file, root_dir=data_source_path,
                                  transform=transforms.ToTensor())


    print('Length of dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            test_x = sample_batched['label'].double().to(device)

            covariates = test_x[:, 2:6]
            test_x = test_x[:, 2:6]
            data_missing = torch.zeros_like(data, dtype=torch.double).to(device)

            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:4]

            covariates_test = test_x * covariates_mask

            covariates_norm = (covariates_test - train_label_means) / train_label_std

            noise_replace = torch.zeros_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data_missing, covariates_norm, covariates_mask)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data,  mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            X_tilde_norm = X_tilde * (1 - covariates_mask) + covariates_norm * covariates_mask
            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates_test * covariates_mask
            covariates_test = X_hat
            Z_pred = predict_scalable(covar_module, likelihoods, train_x, X_hat, pred_mu, zt_list, N, latent_dim, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred, X_tilde_norm)
            [recon_loss_GP, nll_GP] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (nll): ' + str(torch.mean(nll_GP)))
            print('Decoder loss: ' + str(torch.mean(recon_loss_GP)))

            loss = nn.MSELoss(reduction='none')

            se = torch.mul(loss(covariates_test, test_x), 1 - label_mask)
            mask_sum = torch.sum(1 - label_mask, dim=1)
            mask_sum[mask_sum == 0] = 1
            mse_label = torch.sum(se, dim=1) / mask_sum

            print('Label error: ' + str(torch.mean(mse_label)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy(),
                                     torch.mean(nll_GP).cpu().numpy(),
                                     torch.mean(mse_label).cpu().numpy()])
            np.savetxt(os.path.join(results_path, file_num, save_csv), pred_results)

            filename = save_img

            filename = os.path.join(results_path, file_num, filename)

            fig, ax = plt.subplots(5, 10, figsize=(12, 12))
            for i in range(0, 50):
                z1_plot = ax[int(i / 10)][i % 10].imshow(np.reshape(recon_Z[i, :], [36, 36]), cmap='gray')
                ax[int(i / 10)][i % 10].axis('off')

            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')
            return torch.mean(recon_loss_GP), torch.mean(nll_GP)



def MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                      covar_module0, covar_module1, likelihoods, results_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, P, T, id_covariate, varying_T=False, dataset_type='SimpleMNIST'):
    """
    Function to compute Mean Squared Error of test set with GP approximationö

    """
    print("Running tests with a test set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'SimpleMNIST':
            test_dataset = HMNIST_dataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
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


            recon_batch, mu, log_var, X_tilde = nnet_model(data, covariates_norm)

            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates_norm * covariates_mask
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, X_hat, prediction_mu, zt_list, id_covariate, eps=1e-6)
            
            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            np.savetxt(os.path.join(results_path, 'result_error.csv'), pred_results)

def VAEtest(test_dataset, nnet_model, type_nnet, id_covariate):
    """
    Function to compute Mean Squared Error using just a VAE.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label'].to(device)
            data = sample_batched['digit'].to(device)
            mask = sample_batched['mask'].to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)

            if type_nnet == 'rnn':
                recon_batch, mu, log_var = nnet_model(data, covariates, varying_T=True, subjects=label[:, id_covariate])
            else: 
                recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))
