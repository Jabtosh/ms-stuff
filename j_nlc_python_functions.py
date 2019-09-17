import json
import os

from numpy import *
from numpy import linalg as LA
from shutil import copyfile
from mpl_toolkits import axes_grid1
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
# from scipy.optimize import curve_fit
# from scipy.signal import find_peaks_cwt

from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def findpre(sf):
    if '.' in sf:
        return 'B_'
    else:
        return 'Sim_'

def BC_name(BC_PARAM):
    if BC_PARAM==10.:
        return 'radial'
    elif BC_PARAM==11.:
        return 'circular'
    elif BC_PARAM==12.:
        return 'tilted'
    elif BC_PARAM==12.5:
        return 'axial'
    elif 0.<=BC_PARAM and BC_PARAM<10.:
        return 'periodic'
    elif BC_PARAM==20.:
        return 'rectangular'
    else:
        return 'unknown'

def Qfromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_Qxx.dat')):
        QXX = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qxx.dat'), **txt_args)
        print("QXX, ", end='')
        QXY = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qxy.dat'), **txt_args)
        print("QXY, ", end='')
        QYY = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qyy.dat'), **txt_args)
        print("QYY, ", end='')
        QXZ = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qxz.dat'), **txt_args)
        print("QXZ, ", end='')
        QYZ = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qyz.dat'), **txt_args)
        print("QYZ, ", end='')
        QZZ = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_Qzz.dat'), **txt_args)
        print("QZZ")
        return (QXX, QXY, QYY, QXZ, QYZ, QZZ)
    else:
        print('\tQxx.dat not found')
        return (None, None, None, None, None, None)

def QCfromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_cQxx.dat')):
        try:
            QXXC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQxx.dat'), **txt_args)
        except:
            return (None, None, None, None, None, None)
        print("QXXC, ", end='')
        QXYC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQxy.dat'), **txt_args)
        print("QXYC, ", end='')
        QYYC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQyy.dat'), **txt_args)
        print("QYYC, ", end='')
        QXZC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQxz.dat'), **txt_args)
        print("QXZC, ", end='')
        QYZC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQyz.dat'), **txt_args)
        print("QYZC, ", end='')
        QZZC = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_cQzz.dat'), **txt_args)
        print("QZZC")
        if QXXC.ndim==1:
            QXXC = hstack((QXXC,zeros(Axy-len(QXXC))))
            QXYC = hstack((QXYC,zeros(Axy-len(QXYC))))
            QYYC = hstack((QYYC,zeros(Axy-len(QYYC))))
            QXZC = hstack((QXZC,zeros(Axy-len(QXZC))))
            QYZC = hstack((QYZC,zeros(Axy-len(QYZC))))
            QZZC = hstack((QZZC,zeros(Axy-len(QZZC))))
            QXXC = vstack((QXXC, QXXC ))
            QXYC = vstack((QXYC, QXYC ))
            QYYC = vstack((QYYC, QYYC ))
            QXZC = vstack((QXZC, QXZC ))
            QYZC = vstack((QYZC, QYZC ))
            QZZC = vstack((QZZC, QZZC ))
        return (QXXC, QXYC, QYYC, QXZC, QYZC, QZZC)
    else:
        print('\tcQxx.dat not found')
        return (None, None, None, None, None, None)

def Vfromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_VxAvg.dat')):
        VX = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_VxAvg.dat'), **txt_args)
        print("VX, ", end='')
    else:
        print('VxAvg.dat not found')
        VX = None
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_VyAvg.dat')):
        VY = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_VyAvg.dat'), **txt_args)
        print("VY, ", end='')
    else:
        VY = None
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_VzAvg.dat')):
        VZ = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_VzAvg.dat'), **txt_args)
        print("VZ")
    else:
        VZ = None
    return (VX, VY, VZ)

def Rhofromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_density.dat')):
        density = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_density.dat'), **txt_args)
        print("density")
        return density
    else:
        print('density.dat not found')
        return None

def SPfromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_S.dat')):
        SP = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_S.dat'), **txt_args)
        S = SP[:,0]
        P = SP[:,1]
        phi_xy = SP[:,2]
        theta_z = SP[:,3]
        print("S, P, phi_xy, theta_z")
        return SP
    else:
        print('S.dat not found')
        return None

def V_histfromtxt(sf, directory, txt_args):
    prefix = findpre(sf)
    if os.path.exists(os.path.join(directory, f'{prefix:s}{sf:s}_V_hist.dat')):
        V_hist = genfromtxt( os.path.join(directory, f'{prefix:s}{sf:s}_V_hist.dat'), **txt_args)
        return V_hist
    else:
        print('V_hist.dat not found')
        return None

def list_available_data(direc, sf):
    prefix = findpre(sf)
    if "script" in direc:
        suffix = ".h"
    else:
        suffix = ".dat"
    available_data = []
    filecount = 0
    for filename in os.listdir(direc):
        if filename.startswith(f"{prefix:s}") and filename.endswith(suffix) and (not filename[filename.find('_',4)+1] == 'V'):
            if prefix=='Sim_':
                available_data.append(int(filename[4:filename.find('_',4)]))
            else:
                available_data.append((filename[2:filename.find('_',2)]))
    available_data=array(sorted(list(set(available_data))))
    filecount = len(available_data)
    return available_data

def q_of_B(b_list):
    avail_q=[]
    for B in b_list:
        C=B
        A=1-B/3
        avail_q.append(0.5*(B/3/C+sqrt((B/3/C)**2-8*A/3/C)))
    avail_q=array(avail_q)
    return avail_q.round(2)

def pickle_data(directory, directory_script, QX, QY, QZ, Order, S, sf, VX=None, VY=None, VZ=None, Rho=None, SP=None, P=None):
    prefix = findpre(sf)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QX'), QX)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QY'), QY)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QZ'), QZ)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_Order'), Order)
    if S is not None:
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_S'), S)
    if P is not None:
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_P'), P)
    if Rho is not None:
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_density'), Rho)
    if VX is not None:
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_VX'), VX)
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_VY'), VY)
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_VZ'), VZ)
    if SP is not None:
        save(os.path.join(directory_script, f'{prefix:s}{sf:s}_SP'), SP)
    copyfile(os.path.join(directory, f'{prefix:s}{sf:s}_paramts.h'), os.path.join(f'{directory_script}', f'{prefix:s}{sf:s}_paramts.h'))
    print(f'saved {sf:s}')
    
def pickle_Q(directory_script, sf, QXXC, QXYC, QYYC, QXZC, QYZC, QZZC):
    prefix = findpre(sf)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXXC'), QXXC)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QYYC'), QYYC)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QZZC'), QZZC)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXYC'), QXYC)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXZC'), QXZC)
    save(os.path.join(directory_script, f'{prefix:s}{sf:s}_QYZC'), QYZC)
    print(f'saved Q for {sf:s}')

def unpickle_data(sf, directory_script):
    prefix = findpre(sf)
    QX = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QX.npy'))
    QY = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QY.npy'))
    QZ = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QZ.npy'))
    Order = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_Order.npy'))
    if f'{prefix:s}{sf:s}_S.npy' in os.listdir(f'{directory_script}'):
        S = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_S.npy'))
    else:
        S = None
    if f'{prefix:s}{sf:s}_VX.npy' in os.listdir(f'{directory_script}'):
        VX = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_VX.npy'))
        VY = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_VY.npy'))
        VZ = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_VZ.npy'))
    else:
        VX = None
        VY = None
        VZ = None
    if f'{prefix:s}{sf:s}_VX.npy' in os.listdir(f'{directory_script}'):
        density = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_density.npy'))
    else:
        density = None
    if f'{prefix:s}{sf:s}_SP.npy' in os.listdir(f'{directory_script}'):
        SP = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_SP.npy'))
    else:
        SP = None
    return (QX, QY, QZ, Order, S, VX, VY, VZ, density, SP)

def unpickle_Q(sf, directory_script):
    prefix = findpre(sf)
    QXXC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXXC.npy'))
    QYYC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QYYC.npy'))
    QZZC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QZZC.npy'))
    QXYC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXYC.npy'))
    QXZC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QXZC.npy'))
    QYZC = load(os.path.join(directory_script, f'{prefix:s}{sf:s}_QYZC.npy'))
    return (QXXC, QXYC, QYYC, QXZC, QYZC, QZZC)

def read_paramts(sf, direc):
    prefix = findpre(sf)
    with open(os.path.join(direc, f'{prefix:s}{sf:s}_paramts.h'), "r") as text_file:
        lines = text_file.readlines()
    dim = int(lines[2].split('\t')[1])
    ncellsx = int(lines[4].split('\t')[1])
    ncellsy = int(lines[5].split('\t')[1])
    ncellsz = int(lines[6].split('\t')[1])
    dt = float(lines[7].split('\t')[1])
    NSTEPS = int(float(lines[8].split('\t')[1]))
    lapse = int(lines[9].split('\t')[1])
    START_RECORDING = int(float(lines[10].split('\t')[1]))
    
    A_P = (float(lines[11].split('\t')[1]))
    C_P = (float(lines[12].split('\t')[1]))
    GAMMA = (float(lines[13].split('\t')[1]))
    ELCONST = (float(lines[14].split('\t')[1]))
    TUMBLING = (float(lines[15].split('\t')[1]))
    SHEAR_RATE = (float(lines[16].split('\t')[1]))
    FLOW_COUPLING = (float(lines[17].split('\t')[1]))
    
    activity = float(lines[18].split('\t')[1])
    if len(lines)>19:
        nxghost = int(float(lines[19].split('\t')[1]))
    else:
        nxghost = 0
    if len(lines)>20:
        B = float(lines[20].split('\t')[1])
    else:
        B = float(sf)
    if len(lines)>21:
        IC_PARAM = float(lines[21].split('\t')[1])
    else:
        IC_PARAM = -2
        
    if len(lines)>22:
        BC_PARAM = float(lines[22].split('\t')[1])
    else:
        BC_PARAM = -2
    if len(lines)>23:
        CONCENTRATION_PARAM = float(lines[23].split('\t')[1])
    else:
        CONCENTRATION_PARAM = 0.0
    if len(lines)>24:
        W_SURFACE = float(lines[24].split('\t')[1])
    else:
        W_SURFACE = 0.0
    if len(lines)>25:
        nyghost = int(float(lines[25].split('\t')[1]))
    else:
        nyghost = nxghost
    if len(lines)>26:
        ADV_COUPLING = float(lines[26].split('\t')[1])
    else:
        ADV_COUPLING = 0.0
    if len(lines)>27:
        COLLISION_PARAM = int(float(lines[27].split('\t')[1]))
    else:
        COLLISION_PARAM = -1
    if len(lines)>28:
        Q_ACT_LAMBDA = float(lines[28].split('\t')[1])
    else:
        Q_ACT_LAMBDA = 0.0
    if len(lines)>29:
        FSCALE = float(lines[29].split('\t')[1])
    else:
        FSCALE = 1.0
    if len(lines)>30:
        PRESSURE = float(lines[30].split('\t')[1])
    else:
        PRESSURE = 0.0
    if len(lines)>31:
        PSCALE = float(lines[31].split('\t')[1])
    else:
        PSCALE = 1.0
    if len(lines)>32:
        SCL_REF = float(lines[32].split('\t')[1])
    else:
        SCL_REF = 6./10.

    params = dict()
    for line in lines[2:]:
        params[str(line.split('\t')[0][8:])] = float(line.split('\t')[1])

    Axy = ncellsx*ncellsy
    nsteps = int((NSTEPS-START_RECORDING)/lapse)
    X, Y = meshgrid(arange(ncellsx), arange(ncellsy))#for quiver
    X1, Y1 = meshgrid(arange(ncellsx+1), arange(ncellsy+1))#for pcolormesh
    X2, Y2 = meshgrid(arange(0,ncellsx,0.1), arange(0,ncellsy,0.1))#for interpolated pcolormesh
    X = X + 0.5
    Y = Y + 0.5
    return (params, dim, ncellsx, ncellsy, ncellsz, dt, NSTEPS, lapse, START_RECORDING, activity, nxghost, nyghost, Axy, nsteps, \
        X, Y, X1, Y1, X2, Y2, IC_PARAM, BC_PARAM, CONCENTRATION_PARAM, W_SURFACE, ADV_COUPLING, COLLISION_PARAM, A_P, C_P, \
        GAMMA, ELCONST, TUMBLING, SHEAR_RATE, FLOW_COUPLING, B, Q_ACT_LAMBDA, FSCALE, PRESSURE, PSCALE, SCL_REF)

def read_some_parameters(directory, sf):
    prefix = findpre(sf)
    with open(os.path.join(directory, f'{prefix:s}{sf:s}_paramts.h'), "r") as text_file:
        lines = text_file.readlines()
    dim = int(lines[2].split('\t')[1])
    ncellsx = int(lines[4].split('\t')[1])
    ncellsy = int(lines[5].split('\t')[1])
    ncellsz = int(lines[6].split('\t')[1])
    dt = float(lines[7].split('\t')[1])
    NSTEPS = int(float(lines[8].split('\t')[1]))
    lapse = int(lines[9].split('\t')[1])
    START_RECORDING = int(float(lines[10].split('\t')[1]))
    activity = float(lines[18].split('\t')[1])
    if len(lines)>19:
        ncellsghost = float(lines[19].split('\t')[1])
    else:
        ncellsghost = 0
    Axy = ncellsx*ncellsy
    nsteps = int((NSTEPS-START_RECORDING)/lapse)
    X, Y = meshgrid(arange(ncellsx), arange(ncellsy))
    X1, Y1 = meshgrid(arange(ncellsx+1), arange(ncellsy+1))
    X2, Y2 = meshgrid(arange(0,ncellsx,0.1), arange(0,ncellsy,0.1))
    X = X + 0.5
    Y = Y + 0.5
    return (dim, ncellsx, ncellsy, ncellsz, dt, NSTEPS, lapse, START_RECORDING, activity, ncellsghost, Axy, nsteps, X, Y, X1, Y1, X2, Y2)

def otsu(gray, rev=False):
    pixel_number = gray.shape[0]
    mean_weigth = 1.0/pixel_number
    his, bins = histogram(gray, arange(0,257), density=True)
    final_thresh = -1
    final_value = -1
    intensity_arr = arange(256)
    binrange = bins[1:-1]
    if rev:
        binrange = reversed(bins[1:-1])
    for t in binrange: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = sum(his[:t])
        pcf = sum(his[t:])
        if pcb==0 or pcf==0:
            continue
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = sum(intensity_arr[t:]*his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    return final_thresh, final_value
def bidiotsu(gray):
    a1, a2 = otsu(gray, False)
    b1, b2 = otsu(gray, True)
    if b1-a1==1:
        if a2>b2:
            return (a1/256, a1/256)
        else:
            return (b1/256, b1/256)
    else:
        return (a1/256, b1/256)
def seg(intervdist):
    return bidiotsu(intervdist*256)
    
def time_avg(QX, QY, QZ, ncellsx, ncellsy, frac):
    if QX.shape!=QY.shape or QX.shape!=QZ.shape:
        print('shapes dont match!')
        return None
    else:
        tstart = int((1-frac)*QX.shape[0])
        U = QX[tstart:].mean(axis=0)
        V = QY[tstart:].mean(axis=0)
        W = QZ[tstart:].mean(axis=0)
        U = U.reshape( (ncellsy, ncellsx) )
        V = V.reshape( (ncellsy, ncellsx) )
        W = W.reshape( (ncellsy, ncellsx) )
        return (U, V, W)

def director_time_avg(QXXC, QXYC, QYYC, QXZC, QYZC, QZZC, ncellsx, ncellsy, frac):
    tstart = int((1-frac)*QXXC.shape[0])
    qxx1 = QXXC[tstart:].mean(axis=0)
    qyy1 = QYYC[tstart:].mean(axis=0)
    qzz1 = QZZC[tstart:].mean(axis=0)
    qxy1 = QXYC[tstart:].mean(axis=0)
    qxz1 = QXZC[tstart:].mean(axis=0)
    qyz1 = QYZC[tstart:].mean(axis=0)
    Sr = zeros(qxx1.shape)
    nx = zeros(qxx1.shape)
    ny = zeros(qxx1.shape)
    j = 0
    for (qxx,qxy,qyy,qxz,qyz,qzz) in zip(qxx1,qxy1,qyy1,qxz1,qyz1,qzz1):
        Q = array([[qxx, qxy, qxz], [qxy, qyy, qyz], [qxz, qyz, qzz]])
        try:
            eigvals, eigvecs = LA.eig(Q)
        except:
            return ([None,None], None, None)
        ind=argmax(eigvals)
        Sr[j] = eigvals[ind]
        nx[j] = eigvecs[0,ind]
        ny[j] = eigvecs[1,ind]
        #nz[j] = eigvecs[2,ind]
        j += 1
    nx = nx.reshape( (ncellsy, ncellsx) )
    ny = ny.reshape( (ncellsy, ncellsx) )
    Sr = Sr.reshape( (ncellsy, ncellsx) )
    return (nx, ny, Sr)

def pickle_img_data(directory_img, U, V, W, params, name, AvgFrac, Director=True):
    prefix = findpre(name)
    if Director:
        kind = 'Q'
    else:
        kind = 'V'
    params["AvgFrac"] = AvgFrac
    save( os.path.join(directory_img, f'{prefix:s}{name:s}_{kind:s}_img'), (U, V, W, params))
    print('saved img data')

def draw_img_data(U, V, W, params, Director=True, draw_circ=True, vscale=None):
    rcParams.update({'figure.autolayout': True})
    ncellsx = U.shape[1]
    ncellsy = U.shape[0]
    fig, ax = plt.subplots(1,1,figsize=(8*1.3*ncellsx/ncellsy,6.7*1.3), dpi=300)
    ax.set_xlabel("position $x$", fontsize=18)
    ax.set_ylabel("position $y$", fontsize=18)
    X, Y = meshgrid(arange(ncellsx)+0.5, arange(ncellsy)+0.5)
    X1, Y1 = meshgrid(arange(ncellsx+1), arange(ncellsy+1))
    if Director:
        im = ax.pcolormesh(X1, Y1, W, vmin=0, vmax=min([1.2*W[0,0], 1.]), cmap='plasma')
        Q = ax.quiver(X, Y, U, V, color='w', pivot='mid', units='x', scale=1.1, linewidth=.5, width=.1, headlength=0, headwidth = 1)
    else:
        vminmax = max([abs(W).max(), 0.2])
        if vscale is None:
            vscale = max([sqrt(U**2+V**2).max(), 0.04])
        im = ax.pcolormesh(X1, Y1, W, vmin=-vminmax, vmax=vminmax, cmap='bwr')
        Q = ax.quiver(X, Y, U, V, color='k', pivot='mid', units='x', scale=0.25*vscale, linewidth=.5, width=.15, headlength=4.0, headwidth = 2)
    aspect=20
    pad_fraction=0.5
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cbar = im.axes.figure.colorbar(im, cax=cax)
    if draw_circ and (params["BC_PARAM"] < 20 and params["BC_PARAM"] >= 10):
        circle = Ellipse((ncellsx/2, ncellsy/2), ncellsx-2, ncellsy-2, facecolor='none',
                    edgecolor=(0, 0, 0), linewidth=3, alpha=0.8)
        ax.add_patch(circle)
        #fill_polygon = plt.fill(*fill_coords((ncellsx)/2-1,(ncellsy)/2-1), "w")
    ax.axis('equal')
    if Director:
        ax.set_title(f'$\Lambda={params["ACTIVITY"]:f}$, {BC_name(params["BC_PARAM"]):s} boundary conditions')
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel('scalar order parameter $S$', rotation=270, fontsize=18)
    else:
        ax.set_title(f'$\Lambda={params["ACTIVITY"]:f}$, arrow scale={vscale:.2f}, {BC_name(params["BC_PARAM"]):s} boundary conditions')
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('out of plane velocity $v_z$', rotation=270, fontsize=18)
    ax.set_autoscale_on(False)
    ax.set_ylim([0,ncellsy])
    ax.set_xlim([0,ncellsx])
    return fig, ax, im, Q, cbar
    
def draw_streamplot(vx, vy, vz, params, draw_circ=True):
    rcParams.update({'figure.autolayout': True})
    ncellsx = vx.shape[1]
    ncellsy = vx.shape[0]
    fig, ax = plt.subplots(1,1,figsize=(8*1.3*ncellsx/ncellsy,6.7*1.3), dpi=300)
    ax.set_xlabel("position $x$", fontsize=18)
    ax.set_ylabel("position $y$", fontsize=18)
    X, Y = meshgrid(arange(ncellsx)+0.5, arange(ncellsy)+0.5)
    if draw_circ and (params["BC_PARAM"] < 20 and params["BC_PARAM"] >= 10):
        #circle = Ellipse((ncellsx/2, ncellsy/2), ncellsx-2, ncellsy-2, facecolor='none', edgecolor=(0, 0, 0), linewidth=3, alpha=0.8)
        #ax.add_patch(circle)
        fill_polygon = plt.fill(*fill_coords((ncellsx)/2-1,(ncellsy)/2-1), "w")
    strm = ax.streamplot(X, Y, vx, vy, color=sqrt(vx**2+vy**2), linewidth=2, cmap='autumn_r')
    aspect=20
    pad_fraction=0.5
    divider = axes_grid1.make_axes_locatable(ax)
    width = axes_grid1.axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(strm.lines, cax=cax)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.ax.set_ylabel(r'velocity in the $xy$-plane $\sqrt{v_x^2 + v_y^2}$', rotation=270, fontsize=18)
    ax.axis('equal')
    ax.set_autoscale_on(False)
    ax.set_title(f'$\Lambda={params["ACTIVITY"]:f}$, {BC_name(params["BC_PARAM"]):s} boundary conditions')
    ax.set_ylim([0,ncellsy])
    ax.set_xlim([0,ncellsx])
    return fig, ax

def unpickle_img_data(directory_img, name, Director=True):
    prefix = findpre(name)
    if Director:
        kind = 'Q'
    else:
        kind = 'V'
    (U, V, W, params) = load(os.path.join(directory_img, f'{prefix:s}{name:s}_{kind:s}_img.npy'))
    return (U, V, W, params)

def cbrt(val):
    return sign(val)*abs(val)**(1/3)

def fill_coords(r1=40, r2=40, ds=0.1):
    theta = arange(0, 2.0*pi+ds, ds)
    x = r1*cos(theta)
    y = r2*sin(theta)
    x2 = (r1 + 1)*1.42*cos(theta)
    y2 = (r2 + 1)*1.42*sin(theta)
    xf = concatenate((x, x2[::-1]))+r1+1
    yf = concatenate((y, y2[::-1]))+r2+1
    return xf, yf

def s_eq(ev):
    if ev==0.:
        return 0.
    B=ev
    C=ev
    A=(1-ev/3)
    if (B/3/C)**2-8*A/3/C>0.:
        return (0.5*(B/3/C+sqrt((B/3/C)**2-8*A/3/C)))
    else:
        return 0.

def determine_averaging_fraction(dt, NSTEPS, START_RECORDING, lapse):
    timeThreshold = 1000
    minFrames = 50
    TotalTime = dt*NSTEPS
    RecordedFrac = (NSTEPS - START_RECORDING)/NSTEPS
    datapoints = (NSTEPS - START_RECORDING)/lapse
    while TotalTime < timeThreshold*1.05:
        timeThreshold *= 0.9
    if TotalTime >= timeThreshold*1.05:
        TimeFrac = (TotalTime - timeThreshold)/TotalTime
        DataFrac = TimeFrac/RecordedFrac #return-value
        if DataFrac > 1.:
            return 1.
        DataFrames = DataFrac*datapoints
        if DataFrames < minFrames:
            return minFrames/datapoints
        else:
            return DataFrac
    return 0.5

def steps_in_file(directory, sf):
    prefix = findpre(sf)
    with open(os.path.join(directory, f'{prefix:s}{sf:s}_cQxx.dat'), "r") as text_file:
        lines = text_file.readlines()
    return len(lines)

def delete_scanned_files(directory, sf, noQglob):
    prefix = findpre(sf)
    if not noQglob:
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qxx.dat'))
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qxy.dat'))
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qyy.dat'))
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qxz.dat'))
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qyz.dat'))
        os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_Qzz.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQxx.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQxy.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQyy.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQxz.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQyz.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_cQzz.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_VxAvg.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_VyAvg.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_VzAvg.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_density.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_S.dat'))
    os.remove(os.path.join(directory, f'{prefix:s}{sf:s}_paramts.h'))

def neighb(ni):
    if ni<3:
        return 1, ni-1
    elif ni==3:
        return 0, 1
    elif ni<7:
        return -1, 5-ni
    elif ni==7:
        return 0, -1
    elif ni==8:
        return 0, 0

def neighb2(ni):
    if ni<3:
        return 1, ni-1
    elif ni==3:
        return 0, 1
    elif ni<7:
        return -1, 5-ni
    elif ni==7:
        return 0, -1
    elif ni==8:
        return 0, 0
    elif ni<14:
        return 2, ni-11
    elif ni<18:
        return 15-ni, 2
    elif ni<22:
        return -2, 19-ni
    else:
        return ni-23, -2

def smooth_out(line, window_len=210):
    '''smooth out a function through convolution'''
    if window_len==0:
        return line
    sig=r_[line[window_len-1:0:-1],line,line[-2:-window_len-1:-1]]
    wd=eval('hanning'+'(window_len)')
    line2=convolve(wd/wd.sum(),sig,mode='valid')
    line2=line2[int((len(line2)-len(line))/2):int((len(line2)-len(line))/2)+len(line)]
    return line2

def locate_defects(Order, ncellsx, ncellsy, frame, threshold=0.3):
    '''get the positions of all defects as a list of tuples'''
    line = Order[frame]
    plane = Order[frame].reshape( (ncellsy, ncellsx) )
    rel_mins = argrelextrema(line, less)[0]
    abs_mins = where(line < threshold)[0]
    candidates_1d = (set(rel_mins) & set(abs_mins))
    nlist = zeros(9)
    xlocs = []
    ylocs = []
    for lind in candidates_1d:
        x=int(lind%ncellsx)
        y=int(lind/ncellsx)
        moving = True
        while moving:
            for ni in range(9):
                Dx, Dy = neighb(ni)
                nlist[ni] = plane[y+Dy, x+Dx]
            nim = argmin(nlist)
            Dx, Dy = neighb(nim)
            y, x = y+Dy, x+Dx
            if nim==8:
                moving=False
        else:
            xlocs.append(x)
            ylocs.append(y)
    return list(set([(y,x) for y,x in zip(ylocs,xlocs)]))

def locate_defects2(Order, ncellsx, ncellsy, frame, threshold=0.3):
    '''get the positions of all defects as a list of tuples'''
    line = Order[frame]
    plane = Order[frame].reshape( (ncellsy, ncellsx) )
    rel_mins = argrelextrema(line, less)[0]
    abs_mins = where(line < threshold)[0]
    candidates_1d = (set(rel_mins) & set(abs_mins))
    nlist = zeros(9)
    xlocs = []
    ylocs = []
    for lind in candidates_1d:
        x=int(lind%ncellsx)
        y=int(lind/ncellsy)
        moving = True
        co = 0
        while moving:
            if co == 50:
                break
            co += 1
            for ni in range(9):
                Dx, Dy = neighb(ni)
                xn = x + Dx
                yn = y + Dy
                if xn < 0:
                    xn += ncellsx
                elif xn >= ncellsx:
                    xn -= ncellsx
                if yn < 0:
                    yn += ncellsy
                elif yn >= ncellsy:
                    yn -= ncellsy
                nlist[ni] = plane[yn, xn]
            nim = argmin(nlist)
            Dx, Dy = neighb(nim)
            y, x = y+Dy, x+Dx
            if x < 0:
                x += ncellsx
            elif x >= ncellsx:
                x -= ncellsx
            if y < 0:
                y += ncellsy
            elif y >= ncellsy:
                y -= ncellsy
            if nim==8:
                moving=False
        else:
            xlocs.append(x)
            ylocs.append(y)
    return list(set([(y,x) for y,x in zip(ylocs,xlocs)]))

def defect_dist_from_center(coords, ncellsx, ncellsy):
    y,x = coords
    return sqrt((x-(ncellsx/2-.5))**2+(y-(ncellsy/2-.5))**2)

def defect_polar_coords(coords, ncellsx, ncellsy):
    '''transform simulation coordinates to polar'''
    y,x = coords
    x = (x-(ncellsx/2+.5))
    y = (y-(ncellsy/2+.5))
    r = sqrt(x**2 + y**2)
    phi = arctan2(y, x)
    return r, phi

def inverse_mod(angs, threshold=0.5):
    '''inverse function of %(2*pi)'''
    angdiff = diff(angs)
    cuts = where(abs(angdiff) > threshold)[0]
    itrs = 0
    while len(cuts)>0 and itrs<5:
        for cutnum, cut in enumerate(cuts):
            signchange = -sign(angdiff[cut])
            angs = hstack((angs[:cut+1], signchange*pi+angs[cut+1:]))
        angdiff = diff(angs)
        cuts = where(abs(angdiff) > threshold)[0]
        itrs += 1
    return angs

def match_defects_gen_assignment(angs1, angs2, threshold=0.2):
    '''given two arrays of defect locations, return one array of locations for each defect and the assignment vector'''
    phis1 = zeros(len(angs1))
    phis2 = zeros(len(angs1))
    swapindices = zeros(len(angs1))
    other = 0
    for i in range(1, len(angs1)):
        #if (abs(angs1[i]-angs1[i-1]) > threshold) != (other == 1):
        if (abs(angs1[i]-angs1[i-1]) > abs(angs2[i]-angs1[i-1]) and abs(angs1[i]-angs1[i-1]-pi) > abs(angs2[i]-angs1[i-1])) != (other == 1):
            swapindices[i] = 1
            other = 1
        else:
            other = 0
    for i in range(len(angs1)):
        if swapindices[i]:
            phis1[i] = angs2[i]
            phis2[i] = angs1[i]
        else:
            phis1[i] = angs1[i]
            phis2[i] = angs2[i]
    return phis1, phis2, swapindices

def match_defects(angs1, angs2, swapindices):
    '''given two arrays of defect locations and an assignment vector, return one array of locations for each defect'''
    phis1 = zeros(len(angs1))
    phis2 = zeros(len(angs1))
    for i in range(len(angs1)):
        if swapindices[i]:
            phis1[i] = angs2[i]
            phis2[i] = angs1[i]
        else:
            phis1[i] = angs1[i]
            phis2[i] = angs2[i]
    return phis1, phis2

def polar_defect_dynamics(QX, QY, Order, ncellsx, ncellsy, nsteps, sm=50):
    '''return kinematic properties of each defect and of their average'''
    tstep = arange(nsteps)
    dists = zeros(nsteps)
    dists1 = zeros(nsteps)
    dists2 = zeros(nsteps)
    angs = zeros(nsteps)
    angs1 = zeros(nsteps)
    angs2 = zeros(nsteps)
    ori = zeros(nsteps)
    ori1 = zeros(nsteps)
    ori2 = zeros(nsteps)
    now2 = 0
    for i in tstep:
        loc2d = locate_defects(Order, ncellsx, ncellsy, i)
        if len(loc2d)==2:
            Nx = QX[i].reshape( (ncellsy, ncellsx) )
            Ny = QY[i].reshape( (ncellsy, ncellsx) )
            r1, phi1 = defect_polar_coords(loc2d[0], ncellsx, ncellsy)
            r2, phi2 = defect_polar_coords(loc2d[1], ncellsx, ncellsy)
            dists[i] = (r1+r2)*0.5
            dists1[i] = r1
            dists2[i] = r2
            angs[i] = (phi1+phi2)*0.5
            angs1[i] = phi1
            angs2[i] = phi2
            
            alignment = zeros(25)
            x, y = loc2d[0][1], loc2d[0][0]
            for ni in range(25):
                if ni == 8:
                    continue
                Dx, Dy = neighb2(ni)
                alignment[ni] = 1/sqrt(Dx*Dx+Dy*Dy)*abs(Nx[y+Dy, x+Dx]*Dx+Ny[y+Dy, x+Dx]*Dy)
            max_ni = argmax(alignment)
            Dx, Dy = neighb2(max_ni)
            ori1[i] = arccos(1/sqrt(Dx*Dx+Dy*Dy)*abs(cos(phi1)*Dx+sin(phi1)*Dy))
            
            alignment = zeros(25)
            x, y = loc2d[1][1], loc2d[1][0]
            for ni in range(25):
                if ni == 8:
                    continue
                Dx, Dy = neighb2(ni)
                alignment[ni] = 1/sqrt(Dx*Dx+Dy*Dy)*abs(Nx[y+Dy, x+Dx]*Dx+Ny[y+Dy, x+Dx]*Dy)
            max_ni = argmax(alignment)
            Dx, Dy = neighb2(max_ni)
            ori2[i] = arccos(1/sqrt(Dx*Dx+Dy*Dy)*abs(cos(phi2)*Dx+sin(phi2)*Dy))
            
            ori[i] = (ori1[i]+ori2[i])*0.5
        else:
            now2 = i + 1
    tstep = tstep[now2:]
    dists = dists[now2:]
    dists1 = dists1[now2:]
    dists2 = dists2[now2:]
    angs = angs[now2:]
    angs1 = angs1[now2:]
    angs2 = angs2[now2:]
    ori = ori[now2:]
    ori1 = ori1[now2:]
    ori2 = ori2[now2:]
    if angs.size==0:
        return [None]*18
    
    phis1, phis2, swapindices = match_defects_gen_assignment(angs1, angs2)
    dists1, dists2 = match_defects(dists1, dists2, swapindices)
    ori1, ori2 = match_defects(ori1, ori2, swapindices)
    phis1 = inverse_mod(phis1)
    phis2 = inverse_mod(phis2)
    angs = inverse_mod(angs)
    
    tstepvel = tstep[:-1-sm]+0.5
    vel = abs((diff(smooth_out(angs, sm))*smooth_out(dists[:-1], sm)))[:-sm]
    vel1 = abs((diff(smooth_out(phis1, sm))*smooth_out(dists1[:-1], sm)))[:-sm]
    vel2 = abs((diff(smooth_out(phis2, sm))*smooth_out(dists2[:-1], sm)))[:-sm]
    velrad = abs(( diff(smooth_out(dists, sm)) ))[:-sm]
    velrad1 = abs(( diff(smooth_out(dists1, sm)) ))[:-sm]
    velrad2 = abs(( diff(smooth_out(dists2, sm)) ))[:-sm]
    return now2, tstep, dists, dists1, dists2, angs, phis1, phis2, tstepvel, vel, vel1, vel2, velrad, velrad1, velrad2, ori, ori1, ori2

def S_to_cons_json(sf, S_, db_):
    with open(db_, 'r') as fj:
        data = json.load(fj)
    data[str(sf)] = [round(S_[-1], 4)]
    with open(db_, 'w') as fj:
        json.dump(data, fj, sort_keys=True, indent=4)

def findy(x, grid):
    '''for primitive location of defects'''
    ygr = grid[:,x]
    mins = argrelextrema(ygr, less)[0]
    min0 = argmin(ygr[mins])
    min0 = mins[min0]
    return min0

def velocity_analysis(Vx, Vy, Vz, ncellsx, ncellsy, nsteps, analyze_frac=0.1):
    '''likely obsolete'''
    #ux along x-axis
    ux = zeros(ncellsx)
    #uy along y-axis
    uy = zeros(ncellsy)
    #uz along x,y axis
    uz_x = zeros(ncellsx)
    uz_y = zeros(ncellsy)
    #uz of radius
    uz_ = []#zeros(ncellsx*ncellsy*(int(nsteps) - int(nsteps*0.8)))
    uzdist_ = []

    counter = int(nsteps) - int(nsteps*(1.-analyze_frac))
    for i in range(int(nsteps*(1.-analyze_frac)), int(nsteps)-1):
        U = Vx[i].reshape( (ncellsy, ncellsx) )
        V = Vy[i].reshape( (ncellsy, ncellsx) )
        W = Vz[i].reshape( (ncellsy, ncellsx) )

        for x in range(ncellsx):
            ux[x] += 0.5*(U[int(ncellsy/2), x]+ U[int(ncellsy/2)+1, x])
            uz_x[x] += 0.5*(W[int(ncellsy/2), x]+ U[int(ncellsy/2)+1, x])
        for y in range(ncellsy):
            uy[y] += 0.5*(U[y, int(ncellsx/2)]+ U[y, int(ncellsx/2)+1])
            uz_y[y] += 0.5*(W[y, int(ncellsx/2)]+ U[y, int(ncellsx/2)+1])
        for x in range(ncellsx):
            for y in range(ncellsy):
                uzdist_.append(sqrt((x-ncellsx/2)**2+(y-ncellsy/2)**2))
                uz_.append(W[y,x])

    uz_ = array(uz_)
    uzdist_ = array(uzdist_)
    uz = zeros(len(unique(uzdist_)))
    uzdist = zeros(len(unique(uzdist_)))
    for i,pix in enumerate(unique(uzdist_)):
        uzdist[i] = pix
        uz[i] = mean(uz_[uzdist_==pix])
    cellradius = 0.5*(ncellsx)-1
    uz = uz[uzdist<cellradius]
    uzdist = uzdist[uzdist<cellradius]
    ux /= counter
    uy /= counter
    uz_x /= counter
    uz_y /= counter
    fig, (ax, ax2) = plt.subplots(1,2, sharey=True, dpi=90, figsize=(9,4))
    ax.plot(ux)
    ax.plot(uy)
    ax.plot(uz_x)
    ax.plot(uz_y)
    ax.set_xlim([0,max([ncellsx,ncellsy])])
    ax2.plot(uzdist, uz)
    plt.show()

def plot_defect_delta(ncellsx, ncellsy, nsteps, threshneighborhood=3):
    '''plot order parameter as a function of the distance from center'''
    xr = arange(ncellsx)
    i = nsteps-1
    grid = Order[i].reshape( (ncellsy, ncellsx) )
    deltas = []
    orderofdelta = []
    for x in range(ncellsx):
        for y in range(ncellsy):
            deltas.append(sqrt((x-(ncellsx/2-.5))**2+(y-(ncellsy/2-.5))**2))
            orderofdelta.append(grid[y,x])
    mask = (array(deltas)<(ncellsx-2)/2)
    deltas = array(deltas)[mask]
    orderofdelta = array(orderofdelta)[mask]
    deltas /= (ncellsx-2)/2
    average = mean(orderofdelta[deltas<0.8])
    thr = average - threshneighborhood*std(orderofdelta[deltas<0.8])

    fig, ax = plt.subplots(figsize=(8,3),dpi=150)
    ax.scatter(deltas, orderofdelta)
    ax.set_xlim([0,1])
    ax.set_ylim([0,2./3])
    ax.plot([0,1], [thr, thr])
    left, right = seg(deltas[orderofdelta<thr])
    if left==right:
        darray = deltas[orderofdelta<thr]
        delta = mean(darray)
        deltam = median(darray)
        print(f'{delta:.3f} {deltam:.3f}')
        ax.plot([delta, delta], [0, 2./3])
    else:
        darray1 = deltas[(orderofdelta<thr) & (deltas<left)]
        darray2 = deltas[(orderofdelta<thr) & (deltas>right)]
        delta1 = mean(darray1)
        delta1m = median(darray1)
        delta2 = mean(darray2)
        delta2m = median(darray2)
        print(f'{delta1:.3f} {delta1m:.3f}')
        print(f'{delta2:.3f} {delta2m:.3f}')
        ax.plot([delta1, delta1], [0, 2./3])
        ax.plot([delta2, delta2], [0, 2./3])
    return fig, ax

def extract_sims_and_sort_by_activity(directory_script, startatbound, endbeforebound):
    '''takes integer list of simulation names. extracts ones within bounds. returns string list sorted by activity'''
    available_data = array(list_available_data(f"{directory_script}", '1'))
    Sims = available_data[(available_data>=startatbound) & (available_data<endbeforebound)]
    SimsDict = dict()
    for sfint in Sims:
        sf = str(sfint)
        params, dim, ncellsx, ncellsy, ncellsz, dt, NSTEPS, lapse, START_RECORDING, activity, nxghost, nyghost, Axy, nsteps, X, Y, X1, Y1, X2, Y2, \
                IC_PARAM, BC_PARAM, CONCENTRATION_PARAM, W_SURFACE, ADV_COUPLING, COLLISION_PARAM, A_P, C_P, GAMMA, ELCONST, TUMBLING, \
                SHEAR_RATE, FLOW_COUPLING, B, Q_ACT_LAMBDA, FSCALE, PRESSURE, PSCALE, SCL_REF = read_paramts(sf, f'{directory_script}')
        SimsDict[sf]=activity
    return sorted(SimsDict, key=SimsDict.__getitem__)

def d_line(xp, yp, m, b):
    '''project (yp, xp) onto the line given by y=m*x+b. find the point xsol on the line closest to it. return it with its distance from the point.'''
    xsol = (xp+m*yp-m*b)/(1+m**2)
    return sqrt((xp-xsol)**2+(yp-m*xsol-b)**2), xsol

def ell_defect_dynamics(Order, ncellsx, ncellsy, nsteps, sm=50):
    '''return kinematic properties of each defect and of their average'''
    tstep = arange(nsteps)
    dists = zeros(nsteps)
    angs = zeros(nsteps)
    x1 = zeros(nsteps)
    y1 = zeros(nsteps)
    x2 = zeros(nsteps)
    y2 = zeros(nsteps)
    dists1 = zeros(nsteps)
    dists2 = zeros(nsteps)
    angs1 = zeros(nsteps)
    angs2 = zeros(nsteps)
    now2 = 0
    for i in tstep:
        loc2d = locate_defects(Order, ncellsx, ncellsy, i)
        if len(loc2d)==2:
            r1, phi1 = defect_polar_coords(loc2d[0], ncellsx, ncellsy)
            r2, phi2 = defect_polar_coords(loc2d[1], ncellsx, ncellsy)
            dists[i] = (r1+r2)*0.5
            angs[i] = (phi1+phi2)*0.5
            dists1[i] = r1
            dists2[i] = r2
            angs1[i] = phi1
            angs2[i] = phi2
            y1[i] = loc2d[0][0]
            x1[i] = loc2d[0][1]
            y2[i] = loc2d[1][0]
            x2[i] = loc2d[1][1]
        else:
            now2 = i + 1
    tstep = tstep[now2:]
    dists = dists[now2:]
    angs = angs[now2:]
    dists1 = dists1[now2:]
    dists2 = dists2[now2:]
    angs1 = angs1[now2:]
    angs2 = angs2[now2:]
    x1 = x1[now2:]
    x2 = x2[now2:]
    y1 = y1[now2:]
    y2 = y2[now2:]
    if angs.size==0:
        return [None]*19
    swapindices = cart_gen_assignment(x1, y1, x2, y2)
    phis1, phis2 = match_defects(angs1, angs2, swapindices)
    x1, x2 = match_defects(x1, x2, swapindices)
    y1, y2 = match_defects(y1, y2, swapindices)
    dists1, dists2 = match_defects(dists1, dists2, swapindices)
    phis1 = inverse_mod(phis1)
    phis2 = inverse_mod(phis2)
    angs = inverse_mod(angs)
    
    tstepvel = tstep[sm:-sm-1]+0.5
    vel = abs((diff(smooth_out(angs, sm))*smooth_out(dists[:-1], sm))[sm:-sm])
    vel1 = abs((diff(smooth_out(phis1, sm))*smooth_out(dists1[:-1], sm))[sm:-sm])
    vel2 = abs((diff(smooth_out(phis2, sm))*smooth_out(dists2[:-1], sm))[sm:-sm])
    velrad = abs(( diff(smooth_out(dists, sm)) )[sm:-sm])
    velrad1 = abs(( diff(smooth_out(dists1, sm)) )[sm:-sm])
    velrad2 = abs(( diff(smooth_out(dists2, sm)) )[sm:-sm])
    totvel1 = abs((diff(smooth_out(x1, sm))**2 + diff(smooth_out(y1, sm))**2)[sm:-sm])
    totvel2 = abs((diff(smooth_out(x2, sm))**2 + diff(smooth_out(y2, sm))**2)[sm:-sm])
    
    return now2, tstep, dists, dists1, dists2, angs, phis1, phis2, tstepvel, vel, vel1, vel2, velrad, velrad1, velrad2, x1, y1, x2, y2

def cart_gen_assignment(x1, y1, x2, y2):
    swapindices = zeros(len(x1))
    other = 0
    for i in range(1, len(swapindices)):
        if ((x1[i]-x1[i-1])**2+(y1[i]-y1[i-1])**2 > (x2[i]-x1[i-1])**2+(y2[i]-y1[i-1])**2) != (other == 1):
            swapindices[i] = 1
            other = 1
        else:
            other = 0
    return swapindices
    
    return x1, y1, x2, y2, swapindices
def match_defects_gen_assignment(angs1, angs2, threshold=0.2):
    '''given two arrays of defect locations, return one array of locations for each defect and the assignment vector'''
    phis1 = zeros(len(angs1))
    phis2 = zeros(len(angs1))
    swapindices = zeros(len(angs1))
    other = 0
    for i in range(1, len(angs1)):
        #if (abs(angs1[i]-angs1[i-1]) > threshold) != (other == 1):
        if (abs(angs1[i]-angs1[i-1]) > abs(angs2[i]-angs1[i-1]) and abs(angs1[i]-angs1[i-1]-pi) > abs(angs2[i]-angs1[i-1])) != (other == 1):
            swapindices[i] = 1
            other = 1
        else:
            other = 0
    for i in range(len(angs1)):
        if swapindices[i]:
            phis1[i] = angs2[i]
            phis2[i] = angs1[i]
        else:
            phis1[i] = angs1[i]
            phis2[i] = angs2[i]
    return phis1, phis2, swapindices