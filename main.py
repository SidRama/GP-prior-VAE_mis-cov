import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch
from timeit import default_timer as timer
from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE
from dataset_def import HMNIST_dataset
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test_simple_batch
from parse_model_args import ModelArgs
from training import hensman_training_impute

eps = 1e-6

if __name__ == "__main__":
    """
    Root file for running L-VAE.

    Run command: python LVAE.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert not (hensman and mini_batch)
    assert loss_function == 'mse' or loss_function == 'nll', ("Unknown loss function " + loss_function)
    assert not varying_T or hensman, "varying_T can't be used without hensman"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    # set up dataset
    if type_nnet == 'conv':
        if dataset_type == 'SimpleMNIST':
            dataset = HMNIST_dataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                     mask_file=mask_file, root_dir=data_source_path,
                                     transform=transforms.ToTensor())

    # Set up prediction dataset
    if run_tests or generate_images:
            prediction_dataset = HMNIST_dataset(csv_file_data=csv_file_prediction_data,
                                                csv_file_label=csv_file_prediction_label,
                                                mask_file=prediction_mask_file, root_dir=data_source_path,
                                                transform=transforms.ToTensor())
    else:
        prediction_dataset = None

    # Set up dataset for image generation
    if generate_images:
        if type_nnet == 'conv':
            if dataset_type == 'SimpleMNIST':
                generation_dataset = HMNIST_dataset(csv_file_data=csv_file_generation_data, csv_file_label=csv_file_generation_label,
                                                    mask_file=generation_mask_file, root_dir=data_source_path,
                                                    transform=transforms.ToTensor())
    else:
        generation_dataset = None

    # Set up validation dataset
    if run_validation:
        if dataset_type == 'SimpleMNIST':
            validation_dataset = HMNIST_dataset(csv_file_data=csv_file_validation_data,
                                                csv_file_label=csv_file_validation_label,
                                                mask_file=validation_mask_file, root_dir=data_source_path,
                                                transform=transforms.ToTensor())
    else:
        validation_dataset = None

    print('Length of dataset:  {}'.format(len(dataset)))
    N = len(dataset)

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)

    Q = len(dataset[0]['label'])

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    dropout_cov = 0

    net_train_loss_arr = np.empty((0, 1))
    adam_param_list = []
    covar_module = []
    likelihoods = []
    mse_train_arr = np.empty((0, 1))
    mse_test_arr = np.empty((0, 1))

    labels_temp = pd.read_csv(os.path.join(data_source_path, csv_file_label_unmasked), header=0)
    labels = labels_temp.iloc[:, labels_idx]
    labels = torch.Tensor(np.nan_to_num(np.array(labels)))
    time_age_train = labels[:, 3]

    labels_train = labels

    covariates_mean_mse = torch.tensor([])
    covariates_mean_test_mse = torch.tensor([])
    mse_test_best = torch.tensor([float('Inf')])
    mse_train_best = torch.tensor([float('Inf')])
    train_label_means = torch.tensor([])
    train_label_std = torch.tensor([])

    flag = 0

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim, num_dim, vy_init, vy_fixed, X_dim=4).double().to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim, num_dim, vy_init, vy_fixed).to(device)

    # Load pre-trained encoder/decoder parameters if present
    try:
        nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
        print('Loaded pre-trained values.')
    except:
        print('Did not load pre-trained values.')

    nnet_model = nnet_model.double().to(device)

    setup_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    # Get values for GP initialisation:
    Z = torch.zeros(N, latent_dim, dtype=torch.double).to(device)
    train_x = torch.zeros(N, Q, dtype=torch.double).to(device)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(setup_dataloader):
            # no mini-batching. Instead get a batch of dataset size
            label_id = sample_batched['idx']
            train_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)

            label = train_x[:, labels_idx]
            covariates = label
            label_mask = sample_batched['label_mask'].double()
            label_mask = label_mask.to(device)
            covariates_mask = label_mask[:, 0:label.shape[1]]

            train_label_means = torch.sum(covariates, dim=0) / torch.sum(covariates_mask, dim=0)
            covariates_mask_bool = torch.gt(covariates_mask, 0)
            train_label_std = torch.zeros(covariates.shape[1])

            for col in range(0, covariates.shape[1]):
                train_label_std[col] = torch.std(covariates[covariates_mask_bool[:, col], col])

            covariates_norm = (covariates - train_label_means) / train_label_std
            noise_replace = torch.zeros_like(covariates_norm)

            covariates_norm = covariates_norm * covariates_mask
            covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

            recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, covariates_mask)
            std = torch.exp(log_var / 2)

            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates * covariates_mask
            train_x = X_hat

            Z[label_id] = nnet_model.sample_latent(mu, log_var)

    covar_module = []
    covar_module0 = []
    covar_module1 = []
    zt_list = []
    likelihoods = []
    gp_models = []
    adam_param_list = []

    if hensman:
        likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
                                                              noise_constraint=gpytorch.constraints.GreaterThan(
                                                                  1.000E-08)).to(device)

        if constrain_scales:
            likelihoods.noise = 1
            likelihoods.raw_noise.requires_grad = False

        covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                               cat_kernel, bin_kernel, sqexp_kernel,
                                                               cat_int_kernel, bin_int_kernel,
                                                               covariate_missing_val, id_covariate)

        gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                                covar_module0 + covar_module1).to(device)

        # initialise inducing points
        zt_list = torch.zeros(latent_dim, M, train_x.shape[1], dtype=torch.double).to(device)
        for i in range(latent_dim):
            zt_list[i] = train_x[np.random.choice(N, M, replace=False)].clone().detach()
        zt_list.requires_grad_(True)

        adam_param_list.append({'params': covar_module0.parameters()})
        adam_param_list.append({'params': zt_list})

        covar_module0.train().double()
        likelihoods.train().double()

        try:
            gp_model.load_state_dict(
                torch.load(os.path.join(gp_model_folder, 'gp_model.pth'), map_location=torch.device(device)))
            zt_list = torch.load(os.path.join(gp_model_folder, 'zt_list.pth'), map_location=torch.device(device))
        except:
            pass

        m = torch.randn(latent_dim, M, 1).double().to(device).detach()
        H = (torch.randn(latent_dim, M, M) / 10).double().to(device).detach()

        if natural_gradient:
            H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)

        try:
            m = torch.load(os.path.join(gp_model_folder, 'm.pth'), map_location=torch.device(device)).detach()
            H = torch.load(os.path.join(gp_model_folder, 'H.pth'), map_location=torch.device(device)).detach()
        except:
            pass

        if not natural_gradient:
            adam_param_list.append({'params': m})
            adam_param_list.append({'params': H})
            m.requires_grad_(True)
            H.requires_grad_(True)

    else:
        for i in range(0, latent_dim):
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device))

            if constrain_scales:
                likelihoods[i].noise = 1
                likelihoods[i].raw_noise.requires_grad = False

            # set up additive GP prior
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                additive_kernel0, additive_kernel1 = generate_kernel_approx(cat_kernel, bin_kernel, sqexp_kernel,
                                                                            cat_int_kernel, bin_int_kernel,
                                                                            covariate_missing_val, id_covariate)
                covar_module0.append(additive_kernel0.to(device))  # additive kernel without id covariate
                covar_module1.append(additive_kernel1.to(device))  # additive kernel with id covariate
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                              covar_module0[i] + covar_module1[i]).to(device))
                zt = torch.nn.Parameter(z_init.clone().cpu().double().detach(), requires_grad=False)
                zt_list.append(zt)
                adam_param_list.append({'params': covar_module0[i].parameters()})
                adam_param_list.append({'params': covar_module1[i].parameters()})
            else:
                additive_kernel = generate_kernel(cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel,
                                                  covariate_missing_val)
                covar_module.append(additive_kernel.to(device))  # additive kernel GP prior
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                              covar_module[i]).to(device))
                adam_param_list.append({'params': gp_models[i].parameters()})

            gp_models[i].train().double()
            likelihoods[i].train().double()

        for i in range(0, latent_dim):
            gp_model_name = 'gp_model' + str(i) + '.pth'
            zt_list_name = 'zt_list' + str(i) + '.pth'
            try:
                gp_models[i].load_state_dict(
                    torch.load(os.path.join(gp_model_folder, gp_model_name), map_location=torch.device(device)))
                zt_list[i] = torch.load(os.path.join(gp_model_folder, zt_list_name), map_location=torch.device('cpu'))
            except:
                pass

    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)
    nnet_model.train()

    if memory_dbg:
        print("Max memory allocated during initialisation: {:.2f} MBs".format(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        torch.cuda.reset_max_memory_allocated(device)

    if type_KL == 'closed':
        covar_modules = [covar_module]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_modules = [covar_module0, covar_module1]
    #    print(zt_list)
    start = timer()
    if hensman:
        _ = hensman_training_impute(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim,
                                    covar_module0, covar_module1, likelihoods, m, H, zt_list, P, T, varying_T, Q, weight,
                                    id_covariate, loss_function, natural_gradient, natural_gradient_lr,
                                    subjects_per_batch, memory_dbg, eps, file_num, csv_file_validation_data, csv_file_validation_label,
                                    validation_mask_file, data_source_path, csv_file_test_data, csv_file_test_label, test_mask_file,
                                    results_path, validation_dataset,
                                    generation_dataset, prediction_dataset, labels_train)
        m, H = _[5], _[6]
    print("Duration of training: {:.2f} seconds".format(timer() - start))

    if memory_dbg:
        print("Max memory allocated during training: {:.2f} MBs".format(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        torch.cuda.reset_max_memory_allocated(device)

    penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]

    # saving
    print('Saving')
    pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                 os.path.join(save_path, file_num, 'diagnostics.pkl'))

    pd.to_pickle([train_x, mu, log_var, Z, label_id], os.path.join(save_path, file_num, 'plot_values.pkl'))
    torch.save(nnet_model.state_dict(), os.path.join(save_path, file_num, 'final-vae_model.pth'))

    if hensman:
        try:
            torch.save(gp_model.state_dict(), os.path.join(save_path, file_num,  'gp_model.pth'))
            torch.save(zt_list, os.path.join(save_path, file_num,  'zt_list.pth'))
            torch.save(m, os.path.join(save_path, file_num, 'm.pth'))
            torch.save(H, os.path.join(save_path, file_num, 'H.pth'))
        except:
            pass

    else:
        for i in range(0, latent_dim):
            gp_model_name = 'gp_model' + str(i) + '.pth'
            zt_list_name = 'zt_list' + str(i) + '.pth'
            torch.save(gp_models[i].state_dict(), os.path.join(save_path, file_num, gp_model_name))
            try:
                torch.save(zt_list[i], os.path.join(save_path, file_num, zt_list_name))
            except:
                pass

    if memory_dbg:
        print("Max memory allocated during saving and post-processing: {:.2f} MBs".format(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        torch.cuda.reset_max_memory_allocated(device)

    train_Z = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
    prediction_x = torch.zeros(len(dataset), Q - 2, dtype=torch.double).to(device)
    if run_tests or generate_images:
        prediction_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
        full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(len(dataset), Q-2, dtype=torch.double).to(device)

        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(prediction_dataloader):
                label_id = sample_batched['idx']
                data = sample_batched['digit'].double().to(device)
                train_x = sample_batched['label'].double().to(device)
                covariates = train_x[:, labels_idx]

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
                noise_replace = torch.zeros_like(covariates_norm)

                covariates_norm = covariates_norm * covariates_mask
                covariates_norm = (covariates_norm * covariates_mask) + (noise_replace * (1 - covariates_mask))

                recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, covariates_mask)
                X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
                X_hat = X_tilde_denorm * (1 - covariates_mask) + covariates * covariates_mask

                prediction_x[label_id] = X_hat
                train_Z[label_id] = nnet_model.sample_latent(mu, log_var)
                full_mu[label_id] = mu

    # MSE test
    if run_tests:
        with torch.no_grad():
            MSE_test_simple_batch(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, nnet_model,
                                  covar_module0, likelihoods, results_path, file_num, prediction_x, full_mu, zt_list, N,
                                  train_label_means, train_label_std, latent_dim)

    if memory_dbg:
        print("Max memory allocated during tests: {:.2f} MBs".format(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        torch.cuda.reset_max_memory_allocated(device)

