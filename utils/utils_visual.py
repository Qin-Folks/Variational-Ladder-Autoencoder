import numpy as np
import torch
import scipy
import scipy.misc
from torchvision.utils import save_image
import tensorboardX
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from collections import defaultdict
import pandas as pd
from loguru import logger
import os


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def save_tensors_image(filename, inputs, padding=1, pad_value=100, nrow=10):
    # images = image_tensor(inputs, padding)
    inputs = inputs.clamp(0, 1)
    print('inputs in the save: ', inputs.shape)
    return save_image(inputs, filename, padding=padding, pad_value=pad_value, nrow=nrow)


def generate_tensorboard_embeddings(zs, zs_label, save_path=None, spec_targets=None):
    if spec_targets is None:
        # if len(zs_label.shape) > 1:
        #     if zs_label.shape[-1] > 1:
        #         zs_label = np.array(np.min(zs_label, axis=-1) < 0, dtype=int)
        print('len of zs in visualise zs: ', len(zs))
        print('len of zs label in visualise zs: ', len(zs_label))
        writer = tensorboardX.SummaryWriter(save_path)
        writer.add_embedding(zs, metadata=zs_label)
        writer.close()
    else:
        rtn_labels = []
        rtn_data = []
        for a_z, a_z_label in zip(zs, zs_label):
            if -1 in a_z_label:
                for a_spec_target in spec_targets:
                    if a_spec_target in a_z_label:
                        rtn_labels.append(str(a_spec_target))
                        rtn_data.append(a_z)
                        break
            else:
                double_labels = []
                for a_spec_target in spec_targets:
                    if a_spec_target in a_z_label:
                        double_labels.append(a_spec_target)
                if len(double_labels) == len(spec_targets):
                    rtn_labels.append(str(double_labels))
                    rtn_data.append(a_z)
        np_zs = np.array(rtn_data)
        np_labels = np.array(rtn_labels)
        print('len of zs in visualise zs: ', len(np_zs))
        print('len of zs label in visualise zs: ', len(np_labels))
        writer = tensorboardX.SummaryWriter(save_path)
        writer.add_embedding(np_zs, metadata=np_labels)
        writer.close()


def get_label_z_dict(zs, labels):
    rtn = {}
    for a_z, a_label in zip(zs, labels):
        if tuple(a_label) not in rtn.keys():
            rtn[tuple(a_label)] = [np.expand_dims(a_z, 0)]
        else:
            rtn[tuple(a_label)].append(np.expand_dims(a_z, 0))

    for a_key in rtn.keys():
        rtn[a_key] = np.concatenate(rtn[a_key], axis=0)
    return rtn


def plot_misaka_latent_space(zs, labels, save_path, dig_to_use, lat_dim, log):
    misaka_dir = os.path.join(save_path, 'misaka_latent_spaces')
    os.makedirs(misaka_dir)
    label_z_dict = get_label_z_dict(zs, labels)

    for a_key, a_value in label_z_dict.items():
        split_zs = []
        split_label = []
        a_save_path = os.path.join(misaka_dir, str(a_key) + '.png')
        enc_idx = 0
        for _ in dig_to_use:
            split_zs.append(a_value[:, np.r_[enc_idx * lat_dim:(enc_idx + 1) * lat_dim]])
            split_label.append(np.repeat(np.array(enc_idx), len(a_value)))
            enc_idx += 1
        split_zs = np.concatenate(split_zs, 0)
        split_label = np.concatenate(split_label, 0)
        plot_latent_space(split_zs, split_label, a_save_path, log=log)


def plot_latent_space(zs, labels, save_path, log):
    custom_c_map = {}
    custom_c_list = ['red', 'blue', 'orange', 'yellow', 'purple', 'cyan', 'black', 'pink']
    distinct_key = 0
    for a_z, a_label in zip(zs, labels):
        if str(a_label) not in custom_c_map:
            custom_c_map[str(a_label)] = custom_c_list[distinct_key % len(custom_c_list)]
            distinct_key += 1
        plt.scatter(a_z[0], a_z[1], c=custom_c_map[str(a_label)], marker='.', alpha=0.6)
    log.info(custom_c_map)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.clf()


def plot_latent_space_entry(zs, labels, save_path, dig_to_use, lat_dim, **kwargs):
    if 'model_type' in kwargs.keys() and kwargs['model_type'] == 'misaka':
        plot_misaka_latent_space(zs, labels, save_path, dig_to_use, lat_dim, log=kwargs['log'])

    elif 'model_type' in kwargs.keys() and kwargs['model_type'] == 'plain':
        plot_latent_space(zs, labels, save_path, log=kwargs['log'])


def plot_gaussian_circles(loc_list, scale_list, save_path=None, sigma_coe=3, num_to_plot=300, labels=None):
    if len(labels.shape) > 1:
        if labels.shape[-1] > 1:
            labels = np.array(np.min(labels, axis=-1) < 0, dtype=int)

    mu_x_max = -float('inf')
    mu_y_max = -float('inf')

    mu_x_min = float('inf')
    mu_y_min = float('inf')

    color_idx = 0

    rvs = []

    lim_loc_list = loc_list[:num_to_plot]
    lim_scale_list = scale_list[:num_to_plot]
    lim_labels = labels[:num_to_plot]
    for a_mu_, a_sigma_ in zip(lim_loc_list, lim_scale_list):
        a_mu = a_mu_.squeeze()
        a_sigma_ = a_sigma_.squeeze()
        # if not type(a_sigma_) is np.ndarray:
        #     a_sigma_ = a_sigma_.numpy()

        radius = sigma_coe * np.max(a_sigma_)
        a_mu_x = a_mu[0]
        a_mu_y = a_mu[1]

        if (a_mu_x + radius) >= mu_x_max:
            mu_x_max = a_mu_x + radius
        if (a_mu_x - radius) <= mu_x_min:
            mu_x_min = a_mu_x - radius

        if (a_mu_y + radius) >= mu_y_max:
            mu_y_max = a_mu_y + radius
        if (a_mu_y - radius) <= mu_y_min:
            mu_y_min = a_mu_y - radius

        if labels is None:
            rv = plt.Circle(a_mu, radius, fill=False, clip_on=False)
        else:
            colors = cm.rainbow(np.linspace(0, 1, len(set(labels))))
            rv = plt.Circle(a_mu, radius, color=colors[labels[color_idx]], fill=False, clip_on=False)

        rvs.append(rv)
        color_idx = (color_idx + 1)

    fig, ax = plt.subplots()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    axes = plt.gca()
    axes.set_xlim([mu_x_min - 1, mu_x_max + 1])
    axes.set_ylim([mu_y_min - 1, mu_y_max + 1])

    p = PatchCollection(rvs, cmap=cm.jet, alpha=0.4)
    p.set_array(lim_labels)
    ax.add_collection(p)
    fig.colorbar(p, ax=ax)

    plt.scatter(lim_loc_list[:, 0], lim_loc_list[:, 1], c=lim_labels, cmap=cm.jet, marker='D', alpha=0.8, s=2)

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    # if not(labels is None):
    #     # plt.legend(colors, list(range(len(set(labels)))))
    #     pass
    if save_path is None:
        plt.plot()
        plt.show()
        # plt.savefig('plotcircles_test.png')
    else:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


def plot_one_dim_points(values, labels=None):
    values = np.array(values)
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)
    color_dict = defaultdict(str)
    color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'black']

    label_ix = 0
    for a_unique_label in unique_labels:
        color_dict[a_unique_label] = color_list[label_ix]
        label_ix += 1

    ys = np.zeros_like(values) + 0
    for g in unique_labels:
        print('g: ', g)
        ix = np.where(labels == g)
        print(values[ix])
        ax.scatter(values[ix], ys[ix], label=g, c=color_dict[g], alpha=0.7)

    ax.legend()
    plt.show()

