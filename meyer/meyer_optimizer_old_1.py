import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

ranks = np.arange(20)

def init_grid():
    grid = np.ones((20,20), dtype='float')*0.5
    #grid = np.random.uniform(0,1,(21,21))
    for t1 in range(20):
        for t in range(20):
            if t<=t1:
                grid[t,t1] = 1
    return grid

def P(rank):
    if rank in range(20):
        if rank < 14:
            return (2/36)
        else:
            return (1/36)
    else:
        return 0.
def C(rank):
    '''including the given rank'''
    if rank in range(20):
        if rank < 14:
            return (2/36)*(1+rank)
        else:
            return (2/36)*14 + (1/36)*(1+rank-14)
    else:
        return 0.

def H(x, certainty):
    return 0.5*(1+np.tanh(x*certainty))

def get_lie(DOUBT):
    new_LIE = np.zeros((20,20))
    for t1 in ranks[:-1]:
        tsum = 0.
        for t in ranks[t1+1:]:
            new_LIE[t,t1] = 1. - DOUBT[t,t1]
            tsum += 1. - DOUBT[t,t1]
        if tsum < 1e-3:
            norm = 1/(19-t1)
            for t in ranks[t1+1:]:
                new_LIE[t,t1] = norm
        else:
            norm = 1/tsum
            for t in ranks[t1+1:]:
                new_LIE[t,t1] = norm * new_LIE[t,t1]
    t1 = ranks[-1]
    new_LIE[t1,t1] = 1
    return new_LIE

def get_lie_oneup():
    new_LIE = np.zeros((20,20), dtype='float')
    for t1 in ranks[:-1]:
        new_LIE[t1+1,t1] = 1
    new_LIE[ranks[-1],ranks[-1]] = 1
    return new_LIE

def get_mu(DOUBT, LIE):
    #MU = np.ones((20,20))
    MU = DOUBT.copy()
    for t1 in ranks:
        for t in ranks[t1+1:]:
            if 0:# estimation 1
                lie = 18/17*C(t1)
            else:# estimation 2
                pfake = 18/17*C(t1)*LIE[t,t1]
                lie = pfake/(pfake+P(t))
            mu_D = 1+N*(lie-1) # if its me
            mu_T = (N-1)/18. #-(N-1)*C(t)*DOUBT[t+1,t]
            if t == ranks[-1]:
                mu_T -= (N-1) * C(t)# will always be doubted
            for myt in ranks[ranks>t]:
                mu_T += (P(myt) - (N-1) * C(t) * LIE[myt,t]) * DOUBT[myt,t]# above and doubted - below and doubted

                mu_D_N = 1
                mu_T_N = -1/18 # to second order for now
                if myt == ranks[-1]:
                    mu_T_N += C(myt)
                for nt in ranks[ranks>myt]:
                    mu_T_N += (P(nt) + C(myt) * LIE[nt,myt]) * DOUBT[nt,myt]

                    if N == 2:
                        pfake = 18/17*C(myt)*LIE[nt,myt]
                        lie = pfake/(pfake+P(nt))
                        mu_D_N2 = 1+N*(lie-1)
                        mu_T_N2 = (N-1)/18.
                        if nt == ranks[-1]:
                            mu_T_N2 -= (N-1) * C(t)
                        for nnt in ranks[ranks>nt]:
                            mu_T_N2 += (P(nnt) - (N-1) * C(nt) * LIE[nnt,nt]) * DOUBT[nnt,nt]
                            # + p_N3 * outcome_N3; outcome_N3 ~ outcome_N
                    else:
                        mu_D_N2 = 1
                        mu_T_N2 = -1/18
                        if nt == ranks[-1]:
                            mu_T_N2 += C(t)
                        for nnt in ranks[ranks>nt]:
                            mu_T_N2 += (P(nnt) + C(nt) * LIE[nnt,nt]) * DOUBT[nnt,nt]
                            # + p_N3 * outcome_N3; outcome_N3 ~ outcome_N or outcome N_2 depending on number of players
                    outcome_N2 = DOUBT[nt,myt] * mu_D_N2 + (1 - DOUBT[nt,myt]) * mu_T_N2
                    p_N2 = (1 - DOUBT[nt,myt]) * (P(nt) + C(myt) * LIE[nt,myt])
                    mu_T_N += outcome_N2 * p_N2

                outcome_N = DOUBT[myt,t] * mu_D_N + (1 - DOUBT[myt,t]) * mu_T_N
                p_N = (1 - DOUBT[myt,t]) * (P(myt) + C(t) * LIE[myt,t])
                #outcome_N = 0.1 # TEMP
                mu_T += outcome_N * p_N
            MU[t,t1] = mu_D - mu_T
    return MU

def decider(MU, certainty):
    new_DOUBT = np.ones((20,20), dtype='float')
    for t1 in ranks:
        for t in ranks[t1+1:]:
            new_DOUBT[t,t1] = H(MU[t,t1], certainty)
    return new_DOUBT

certainty = 15
player_counts = [2,3,6,9]
N_vars = len(player_counts)
PD_N = np.ones((N_vars,20,20))
PL_N = np.ones((N_vars,20,20))
mu_N = np.ones((N_vars,20,20))
certainty = 2.
for layer, N in enumerate(player_counts):
    DOUBT = init_grid()
    LIE = get_lie_oneup()
    for i in range(50):
        MU = get_mu(DOUBT, LIE)
        DOUBT = decider(MU, certainty)
        LIE = get_lie(DOUBT)
    PD_N[layer,:,:] = DOUBT
    PL_N[layer,:,:] = LIE
    mu_N[layer,:,:] = MU

###############################################################

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
    ax.set_ylabel(last_throw_label)
    ax.set_xlabel(before_last_throw_label)
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