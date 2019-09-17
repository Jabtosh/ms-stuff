# If argument (sf) is given, see if name exists in dir_script. Print out all names present in dir_script.
import sys
import os
import numpy as np

directory_script = '/scratch.local/data/script_output'

from joscha_nlc_python_functions import *

if len(sys.argv) > 1:
	sf = str(sys.argv[1])
	prefix = findpre(sf)
	if os.path.exists(os.path.join(directory_script, f'{prefix}{sf:s}_QX.npy')):
		print(f'YES. {sf} is already inside {directory_script}')
	else:
		print(f'NO. {sf} does not yet exist inside {directory_script}')
	available_data = list_available_data(directory_script, sf)
else:
	sf = '1'
	available_data = list_available_data(directory_script, sf)
available_data = np.array(available_data)
for i in range(90):
	lbound = i * 100
	rbound = (i+1) * 100
	v = available_data[(available_data >= lbound) & (available_data < rbound)]
	if len(v)>0:
		for j in range(len(v)):
			if j>0 and int(v[j]/10)!=int(v[j-1]/10):
				print(' |  ', end='')
			print(f'{v[j]:3d}', end=' ')
		print('')