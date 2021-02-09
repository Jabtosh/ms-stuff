import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def init_doubt():
    grid = np.ones((20,20), dtype='float')*0.5
    for V_m1 in V:
        for E_0 in V:
            if E_0 <= V_m1 or Q[V_m1]/(1-P21) >= 0.5:
                grid[E_0, V_m1] = 1
            else:
                grid[E_0, V_m1] = Q[V_m1]/(1-P21)
    return grid

def init_lie():
    grid = np.zeros((20,20), dtype='float')
    for V_m1 in V:
        for E_0 in V[V_m1+1:]:
            grid[E_0, V_m1] = P[E_0]/(1-Q[V_m1]-P21)
    return grid

def one(V_0, nxtpl):
    return 1.

def calc_T(E_0):
    global B,T
    total = P21
    goes_on = np.zeros(20)
    for V_0 in V[E_0+1:]:
        if B[V_0] >= 0:
            nxt = B[V_0]
        else:
            B[V_0] = calc_B(V_0)
            nxt = B[V_0]
        goes_on[V_0] = (1-eD[V_0, E_0])*nxt
    best_lie = 0#max(goes_on)
    for V_0 in V[E_0+1:]:
        total += P[V_0] * max(( eD[V_0, E_0] + goes_on[V_0] ), (best_lie))
        total += Q[E_0] * L[V_0, E_0] * goes_on[V_0]
    return total

def calc_B(V_0):
    global B,T
    total = 0
    goes_on = np.zeros(20)
    for E_1 in V[V_0+1:]:
        if T[E_1] <= 0:
            T[E_1] = calc_T(E_1)
        goes_on[E_1] = (1-D[E_1, V_0])*T[E_1]
    best_lie = 1#min(goes_on)
    for E_1 in V[V_0+1:]:
        total += P[E_1] * min( (1-D[E_1, V_0])*T[E_1] , best_lie )
        total += Q[V_0] * eL[E_1, V_0] * ( D[E_1, V_0] + (1-D[E_1, V_0])*T[E_1] )
    return total

def calc_T_B():
    global T,B
    T = np.ones(20)*(-1)
    B = np.ones(20)*(-1)
    for i in V[::-1]:
        if T[i] <= 0:
            T[i] = calc_T(i)

def calc_D():
    new_D = init_doubt()
    for V_m1 in V:
        for E_0 in V[V_m1+1:]:
            q = T[E_0]*(1-P21)/(Q[V_m1])
            p = 1/(1+q)
            if p < .5:
                new_D[E_0, V_m1] = p
            else:
                new_D[E_0, V_m1] = 1
    return new_D

V = np.arange(20)
P = np.zeros(20)
P[V <14] = 2/36
P[V>=14] = 1/36
Q = P.cumsum()
P21 = 2/36

D = init_doubt()
eD = init_doubt()
L = init_lie()
eL = init_lie()
T = np.ones(20)*(-1)
B = np.ones(20)*(-1)

#TODO array unnecessaryLie[E_0, V_m1] of probabilities
#when implemented reactivate best_lie
player_counts = [2]
N_vars = len(player_counts)
PD_N = np.ones((N_vars,20,20))
PL_N = np.ones((N_vars,20,20))
mu_N = np.ones((N_vars,20,20))
for layer, N in enumerate(player_counts):
    D = init_doubt()
    eD = init_doubt()
    L = init_lie()
    eL = init_lie()
    for i in range(3):
        calc_T_B()
        D = calc_D()
        eD = D.copy()
        #L = get_lie(DOUBT)
    PD_N[layer,:,:] = D
    PL_N[layer,:,:] = L
    #mu_N[layer,:,:] = MU

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