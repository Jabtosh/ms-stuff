import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


sbplt_adj_args = dict(left=0.06,bottom=0.06,right=0.85,top=0.9,wspace=0.03,hspace=0.40)
last_throw_label = r'$t$'
before_last_throw_label = r'$t_{-1}$'
fig, axar = plt.subplots(int(np.sqrt(N_vars)), int(N_vars/int(np.sqrt(N_vars))), figsize=(11,9),sharex=True,sharey=True)
axar = np.asarray(axar)
#fig.tight_layout()
fig.suptitle('Doubt', fontsize=16)
plt.subplots_adjust(**sbplt_adj_args)
images = []
for i in range(N_vars):
    ax = axar.flatten()[i]
    images.append(ax.imshow(PD_N[i,:,:].round(1), origin='lower'))
    ax.set_xlabel(last_throw_label)
    ax.set_ylabel(before_last_throw_label)
    ax.set_title(f'{player_counts[i]} players')
    #cbar = fig.colorbar(img)
    #cbar.ax.set_ylabel('Should doubt')
cbar = fig.colorbar(images[0], ax=axar.ravel().tolist(), shrink=0.95)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Throw', 'Doubt'])

fig2, axar2 = plt.subplots(int(np.sqrt(N_vars)), int(N_vars/int(np.sqrt(N_vars))), figsize=(11,9),sharex=True,sharey=True)
plt.subplots_adjust(**sbplt_adj_args)
fig2.suptitle('Lie', fontsize=16)
for i in range(N_vars):
    ax = axar2.flatten()[i]
    img = ax.imshow(PL_N[i,:,:], origin='lower')
    ax.set_ylabel(last_throw_label)
    ax.set_xlabel(before_last_throw_label)
    ax.set_title(f'{player_counts[i]} players')
cbar = fig2.colorbar(img, ax=axar2.ravel().tolist(), shrink=0.95)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['dont claim', 'claim'])

fig3, axar3 = plt.subplots(int(np.sqrt(N_vars)), int(N_vars/int(np.sqrt(N_vars))), figsize=(11,9),sharex=True,sharey=True)
plt.subplots_adjust(**sbplt_adj_args)
fig3.suptitle('Expected profit from doubting', fontsize=16)
images = []
for i in range(N_vars):
    ax = axar3.flatten()[i]
    images.append(ax.imshow(mu_N[i,:,:]/player_counts[i], origin='lower', cmap='RdYlGn', vmin=-1, vmax=1))
    ax.set_ylabel(last_throw_label)
    ax.set_xlabel(before_last_throw_label)
    ax.set_title(f'{player_counts[i]} players')
cbar = fig3.colorbar(images[0], ax=axar3.ravel().tolist(), shrink=0.95)
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())
for im in images:
    im.callbacksSM.connect('changed', update)
plt.show()