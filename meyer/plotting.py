import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from constants import L, V
from meyer_optimizer import do_doubt, mu


def generate_plot_grid(data, title, x_label=r'$t_{-1}$', y_label=r'$t$', normalize=False):
    fig, ax_array = plt.subplots(nrows=int(np.sqrt(n_plots)), ncols=math.ceil(n_plots / int(np.sqrt(n_plots))),
                                 sharex=True, sharey=True, figsize=(11, 9))
    ax_array = np.asarray(ax_array)
    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(**subplot_adjustments)
    images = []
    for i_plot, n in enumerate(player_counts):
        ax = ax_array.flatten()[i_plot]
        images.append(ax.imshow(data[i_plot, :, :].round(2), origin='lower', aspect='equal', cmap='GnBu'))
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(f'{n} players')
    color_bar = fig.colorbar(images[0], ax=ax_array.ravel().tolist(), shrink=0.95)
    if normalize:
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
    return fig, color_bar


def plot_do_doubt():
    do_doubt_array = np.zeros((n_plots, L, L))
    for i_plot, n in enumerate(player_counts):
        for claim_m2 in range(L):
            for claim_m1 in range(L):
                do_doubt_array[i_plot, claim_m1, claim_m2] = do_doubt(n, claim_m2, claim_m1, rounds_remaining)
    fig_doubt, color_bar_doubt = generate_plot_grid(do_doubt_array, title=f'Doubt: {rounds_remaining} rounds remaining')
    color_bar_doubt.set_ticks([0, 1])
    color_bar_doubt.set_ticklabels(['Throw', 'Doubt'])


def plot_mu():
    players_until_me = 0
    mu_array = -np.ones((n_plots, L, L))
    for i_plot, n in enumerate(player_counts):
        for claim_m2 in range(L):
            for claim_m1 in V[claim_m2 + 1:]:
                mu_array[i_plot, claim_m1, claim_m2] = mu(n, claim_m2, claim_m1, players_until_me, rounds_remaining)

    fig_mu, color_bar_mu = generate_plot_grid(
        mu_array, title=f'Mu: {rounds_remaining} rounds remaining, {players_until_me} players until me', normalize=True)


if __name__ == '__main__':
    subplot_adjustments = dict(left=0.06, bottom=0.06, right=0.85, top=0.9, wspace=0.03, hspace=0.40)
    rounds_remaining = 0
    player_counts = [2, 3, 4, 5]
    n_plots = len(player_counts)

    plot_mu()
