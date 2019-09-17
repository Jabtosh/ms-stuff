import os
import datetime
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joscha_nlc_python_functions import *

from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

redo_all = False

os.chdir(os.path.dirname(os.path.realpath(__file__)))
if 1:# at mpi
	directory_stationary = '/scratch.local/data/output_stationary'
	directory_stills = '/scratch.local/data/OwnCloud/Stills'
	directory_stills_new = '/scratch.local/data/OwnCloud/Images'
	#directory_stills = '/scratch.local/data/output_stills'
else:# on laptop
	directory_stationary = r'A:\data\output_stationary'
	directory_stills = r'A:\data\output_stills'
suffix = "_img.npy"
Q_data = []
V_data = []
filecount = 0
for filename in os.listdir(directory_stationary):
	if filename.endswith("Q"+suffix) : 
		Q_data.append(str(int(filename[4:filename.find('_',4)])))
	if filename.endswith("V"+suffix) : 
		V_data.append(str(int(filename[4:filename.find('_',4)])))
filecount = len(Q_data)+len(V_data)
print(filecount, 'files fetched')

for sf in V_data:
	vx, vy, vz, params = unpickle_img_data(directory_stationary, sf, Director=False)
	
	if f'{sf:s}_V.png' not in os.listdir(directory_stills) or redo_all:
		fig, ax, pcm, quiv, cbar = draw_img_data(vx, vy, vz, params, Director=False, vscale=None)
		fig.savefig(os.path.join(directory_stills, f'{sf:s}_V.png'), bbox_inches='tight', dpi=300)
		plt.close('all')
	if f'{sf:s}_V_stream.png' not in os.listdir(directory_stills) or redo_all:
		try:
			fig, ax = draw_streamplot(vx, vy, vz, params, draw_circ=True)
			fig.savefig(os.path.join(directory_stills, f'{sf:s}_V_stream.png'), bbox_inches='tight', dpi=300)
			plt.close('all')
		except:
			print(f'failed streamplot for {sf}')
		print(f'finished V for {sf}')
	if int(sf)>=3000 and f'{sf:s}_V.png' not in os.listdir(directory_stills_new):
		fig, ax, pcm, quiv, cbar = draw_img_data(vx, vy, vz, params, Director=False, vscale=None)
		fig.savefig(os.path.join(directory_stills_new, f'{sf:s}_V.png'), bbox_inches='tight', dpi=300)
		plt.close('all')
		try:
			fig, ax = draw_streamplot(vx, vy, vz, params, draw_circ=True)
			fig.savefig(os.path.join(directory_stills_new, f'{sf:s}_V_stream.png'), bbox_inches='tight', dpi=300)
			plt.close('all')
		except:
			print(f'failed streamplot for {sf}')
		print(f'finished V for {sf} in new folder')


for sf in Q_data:
	nx, ny, Sr, params = unpickle_img_data(directory_stationary, sf)

	if f'{sf:s}_Q.png' not in os.listdir(directory_stills) or redo_all:
		fig, ax, pcm, quiv, cbar = draw_img_data(nx, ny, Sr, params)
		fig.savefig(os.path.join(directory_stills, f'{sf:s}_Q.png'), bbox_inches='tight', dpi=300)
		plt.close('all')
		print(f'finished Q for {sf}')

	if int(sf)>=3000 and f'{sf:s}_Q.png' not in os.listdir(directory_stills_new):
		fig, ax, pcm, quiv, cbar = draw_img_data(nx, ny, Sr, params)
		fig.savefig(os.path.join(directory_stills, f'{sf:s}_Q.png'), bbox_inches='tight', dpi=300)
		plt.close('all')
		print(f'finished Q for {sf} in new folder')
