import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.autograd import Variable
import os
import argparse

from datasets import MNIST_13Rot

from load_model import load_model_from_save_dict
from plotting import plot_sample_generations_from_each_cluster_torch_grid


import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_sample_generation_1d(sample_dict, n_clusters, n_examples_per_cluster):
    """
    Creates a scatter plot for 1D generated samples.
    
    Args:
       sample_dict: A dictionary mapping cluster indices (0,...,n_clusters-1) to a list of tuples.
                    Each tuple is (generated_sample, label_str), where generated_sample is a scalar tensor.
       n_clusters:  Number of clusters.
       n_examples_per_cluster: Number of generated examples per cluster.
    
    Returns:
       A matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster in range(n_clusters):
        # Get samples for the given cluster and convert them to numpy scalars.
        samples = np.array([s[0].item() for s in sample_dict.get(cluster, [])])
        if samples.size > 0:
            # Plot each sample as a point along a horizontal line at y = cluster.
            ax.scatter(samples, np.full_like(samples, cluster), color=colors[cluster],
                       alpha=0.7, label=f"Cluster {cluster}")
    
    ax.set_xlabel("Generated Sample Value")
    ax.set_ylabel("Cluster")
    ax.set_title("Generated Samples per Cluster (1D)")
    ax.legend()
    return fig

def plot_training_loss(epochs, losses):
    """
    Plots the training loss over epochs.
    
    Args:
       epochs: List or array of epoch numbers.
       losses: List or array of loss values.
    
    Returns:
       A matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, marker='o', linestyle='-')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss over Epochs")
    ax.grid(True)
    return fig


def eval_sample_generation():
    """
    Run this function to perform post-training sample generation plot of a model.
    For more information on the plot, see Section 4.4 and Appendix E.6 of the paper.
    """
    parser = argparse.ArgumentParser(description='Evaluation parsing.')
    parser.add_argument('--model_path', type=str, default="pretrained_models/mnist.pt", metavar='N', help="Path to a model file of type .pt .")
    parser.add_argument('--results_dir', type=str, default="results/mnist", metavar='N', help="Path to a directory where results will be stored.")
    parser.add_argument('--device', type=str, default='cpu', metavar='N', help="device to use for all heavy tensor operations, e.g. 'cuda:0', 'cpu', ...")
    parser.add_argument('--temperature', type=float, default=0.3, metavar='N', help='temperature factor for scaling covariance matrix of sampling distributions.')
    eval_args, unknown = parser.parse_known_args()

    # configs
    model_path = eval_args.model_path
    results_dir = eval_args.results_dir
    device_string = eval_args.device
    temperature = eval_args.temperature

    # define device and load model
    mfcvae, args = load_model_from_save_dict(model_path, map_location=device_string)

    # changes model to evaluation mode (e.g. dropout, batch norm affected)
    mfcvae.eval()

    # transfer model to device
    args.device = device_string
    device = torch.device(device_string)
    mfcvae.device = device
    mfcvae = mfcvae.to(device)

    if args.dataset == 'mnist_13rot':
        print("Initialize MNIST_13Rot data and data loaders...")
        from datasets import MNIST_13Rot

        train_data = MNIST_13Rot(train=True, device=args.device)
        test_data  = MNIST_13Rot(train=False, device=args.device)

    else:
        exit("Dataset not implemented.")


    # initialize data loaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390

    mfcvae.eval()  # changes model to evaluation mode (e.g. dropout, batch norm affected)

    if args.do_progressive_training:
        epoch = int(sum(args.n_epochs_per_progressive_step)) - 1
    else:
        epoch = args.n_epochs - 1

    vis_examples_per_cluster_logged = [{} for j in range(args.J_n_mixtures)]
    vis_count_examples_per_cluster = [{} for j in range(args.J_n_mixtures)]
    vis_z_j_per_cluster = [{} for j in range(args.J_n_mixtures)]
    index_to_y_j_cluster = {}
    index_to_prob_p_c_j_z_j = {}

    for j in range(args.J_n_mixtures):
        for k in range(mfcvae.n_clusters_j_list[j]):
            vis_examples_per_cluster_logged[j][k] = []
            vis_z_j_per_cluster[j][k] = []
            vis_count_examples_per_cluster[j][k] = 0
    for n in range(len(test_data)):
        index_to_y_j_cluster[n] = []

    row_indices = []
    num_nonempty_clusters = []

    for batch_idx, (x, y_true) in enumerate(test_loader):
        x, y_true = x.to(device), y_true.to(device)
        if args.dataset == 'fast_mnist':
            x = x.view(x.size(0), -1).float()
        global_indices = list(range(batch_idx*args.eval_batch_size, (batch_idx+1)*args.eval_batch_size))
        x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, epoch, 0)
        prob_p_c_j_z_j_list = mfcvae.compute_q_c_j_x(z_sample_q_z_j_x_list)
        for h in range(z_sample_q_z_j_x_list[0].shape[0]):  # is probably == batch size
            g = global_indices[h]
            index_to_prob_p_c_j_z_j[g] = [prob_p_c_j_z_j_list[j][h].detach().cpu() for j in range(args.J_n_mixtures)]

        y_pred_j_list = []
        for j in range(mfcvae.J_n_mixtures):
            prob_p_c_j_z_j_list[j] = prob_p_c_j_z_j_list[j].data.cpu().numpy()
            y_pred_j = np.argmax(prob_p_c_j_z_j_list[j], axis=1)
            y_pred_j_list.append(y_pred_j)

        for j in range(mfcvae.J_n_mixtures):
            for k in range(mfcvae.n_clusters_j_list[j]):
                y_pred = y_pred_j_list[j]
                indices = (np.where(y_pred == k)[0])
                count_indices = indices.shape[0]
                indices = indices.tolist()
                for h in indices:
                    index_to_y_j_cluster[global_indices[h]].append(k)
                vis_count_examples_per_cluster[j][k] += count_indices

    # print("looped through test data.")

    # build a useful data structure to handle the clustering probabilities
    j_to_cluster_to_index_prob = {}
    # create empty things
    for j in range(args.J_n_mixtures):
        j_to_cluster_to_index_prob[j] = {}
        for c in range(args.n_clusters_j_list[j]):
            j_to_cluster_to_index_prob[j][c] = []

    for (index, prob_list) in index_to_prob_p_c_j_z_j.items():
        for j in range(args.J_n_mixtures):
            cluster_j = torch.argmax(prob_list[j])
            cluster_j = cluster_j.item()
            j_to_cluster_to_index_prob[j][cluster_j].append((index, prob_list[j][cluster_j]))

    # Sort clusters s.t. cluster with the largest "average confidence" is 0, second largest 1 etc.
    cluster_average_confidence = {}
    for j in range(args.J_n_mixtures):
        cluster_average_confidence[j] = {}
        for c in range(args.n_clusters_j_list[j]):
            cluster_average_confidence[j][c] = np.nan_to_num(np.mean([j_to_cluster_to_index_prob[j][c][k][1] for k in range(len(j_to_cluster_to_index_prob[j][c]))]))
        # sort
        cluster_index_average_confidence_list = [(cluster_j, score) for (cluster_j, score) in cluster_average_confidence[j].items()]
        cluster_index_average_confidence_list = sorted(cluster_index_average_confidence_list, key=lambda tuple: tuple[1], reverse=True)
        # print(cluster_index_average_confidence_list)
        cluster_j_sorted = [cluster_j for (cluster_j, score) in cluster_index_average_confidence_list]
        row_indices.append(cluster_j_sorted)
        # compute the number of clusters with non-empty assignment from the test set
        num_nonempty_clusters.append(len(np.argwhere(np.array([cluster_index_average_confidence_list[i][1] for i in range(args.n_clusters_j_list[j])]))))
        fromto_mapping = {cluster_j: i for i, cluster_j in enumerate(cluster_j_sorted)}
        # remap the dictionary - https://gist.github.com/pszaflarski/b139736415abbf8d344d77524baaece8
        j_to_cluster_to_index_prob[j] = {fromto_mapping.get(k, k): v for k, v in j_to_cluster_to_index_prob[j].items() if k in fromto_mapping}

    # log sample generations per facet and cluster, in the order of y_pred_j_count_list
    args.n_sample_generations_per_cluster = 10
    # print('Checkpoint 1.')
    fig_list = plot_sample_generation(row_indices, num_nonempty_clusters, mfcvae, args, temperature, results_dir, show_plot=True)

def plot_sample_generation(row_indices, num_clusters, mfcvae, args, temperature, results_dir=None, show_plot=False):
    """
    Args:
       row_indices: The indices which decide the row sorting.
       num_clusters: The number of clusters to be visualised.
       mfcvae: The trained MFCVAE model.
       args: The arguments associated with the training procedure.
       temperature: The multiplier for the variance of p(z|c) during sampling.
       results_dir: Path to save the output plots.
       show_plot: Whether to show the plots by plt.show().
    """
    if args.dataset == 'mnist_13rot':
        in_channels = 1
        width, height = 28, 28
    else:
        exit("dataset not known")


    

    

    vis_sample_generations_logged = {}
    for i in range(mfcvae.J_n_mixtures):  # J clusterings
        vis_sample_generations_logged[i] = {}
        for j in range(mfcvae.n_clusters_j_list[i]):
            vis_sample_generations_logged[i][j] = []

    fig_list = []
    for i in range(mfcvae.J_n_mixtures):
        # sort rows by "average confidence":
        y_pred_j_confidence_descend_index = row_indices[i]
        for j0 in range(num_clusters[i]):
            # sort rows by "average confidence":
            j = y_pred_j_confidence_descend_index[j0]
            for K in range(args.n_sample_generations_per_cluster):
                z_sample_list = []
                for k in range(args.J_n_mixtures):
                    c_k = int(D.Categorical(probs=mfcvae.pi_p_c_j_list[k]).sample())
                    if args.cov_type_p_z_c == 'diag':
                        z_sample_list.append(
                            torch.unsqueeze(D.Normal(loc=mfcvae.mu_p_z_j_c_j_list[k][:, c_k],
                                                     scale=temperature * mfcvae.sigma_square_p_z_j_c_j_list[
                                                               k][:, c_k]).sample(), 0))
                    elif args.cov_type_p_z_c == 'full':
                        z_sample_list.append(torch.unsqueeze(
                            D.MultivariateNormal(loc=mfcvae.mu_p_z_j_c_j_list[k][:, c_k],
                                                 scale_tril=temperature * mfcvae.l_mat_p_z_j_c_j_list[k][:,
                                                            :, c_k]).sample(), 0))
                cluster_mu = mfcvae.mu_p_z_j_c_j_list[i][:, j]
                if args.cov_type_p_z_c == 'diag':
                    cluster_sigma_square = mfcvae.sigma_square_p_z_j_c_j_list[i][:, j]
                    p_z_i_c_i = D.Normal(loc=cluster_mu, scale=temperature*cluster_sigma_square)
                elif args.cov_type_p_z_c == 'full':
                    cluster_l_mat = mfcvae.l_mat_p_z_j_c_j_list[i][:, :, j]
                    p_z_i_c_i = D.MultivariateNormal(loc=cluster_mu, scale_tril=temperature*cluster_l_mat)
                z_sample_list[i] = torch.unsqueeze(p_z_i_c_i.sample(), 0)
                x_generated_samples = mfcvae.decode(
                    z_sample_q_z_j_x_list=z_sample_list)  # slightly inconcistent naming
                x_generated_samples = torch.squeeze(x_generated_samples, dim=0)
                x_generated_samples = torch.clamp(x_generated_samples, min=1e-10,
                                                  max=1 - (1e-10))



                if args.dataset == "simple_2d_gmm":
                    vis_sample_generations_logged[i][j0] = vis_sample_generations_logged[i][j0] + [
                        (x_generated_samples.view(-1).cpu().detach(),
                         'input_facet_' + str(i) + '_pred_' + str(j))]
                    
                else:
                    vis_sample_generations_logged[i][j0] = vis_sample_generations_logged[i][j0] + [
                        (x_generated_samples.view(in_channels, width, height).cpu().detach(),
                         'input_facet_' + str(i) + '_pred_' + str(j))]

                

        # do plotting
        fig = plot_sample_generations_from_each_cluster_torch_grid(
            sample_dict=vis_sample_generations_logged[i],
            n_clusters=num_clusters[i],
            n_examples_per_cluster=args.n_sample_generations_per_cluster)
        fig_list.append(fig)
        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, 'generations_facet_%d.pdf'%(i)), format='pdf')  # , dpi=3000
        if show_plot:
            plt.show()
        plt.close(fig)

    # Now create a 1D plot for the generated samples
    if args.dataset == "1d":
        print(100*"----")
        sample_fig = plot_sample_generation_1d(
            vis_sample_generations_logged[0],
            n_clusters=args.n_clusters_j_list[0],
            n_examples_per_cluster=args.n_sample_generations_per_cluster
        )
        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, 'generations_facet_0.pdf'), format='pdf')
        wandb.log({"Generated Samples (1D)": wandb.Image(sample_fig)})
        plt.close(sample_fig)
    else:
        # (Handle other datasets if needed.)
        pass

    print("sample generation done.")
    return fig_list

if __name__ == '__main__':
    eval_sample_generation()
