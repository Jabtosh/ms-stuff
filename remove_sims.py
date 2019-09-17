# Takes any number of names (sf) as argument. Deletes all associated files from all involved directories.
import sys
import os
from joscha_nlc_python_functions import *

delete_state_files = True


def remove_file(curdir, filename, count=-1):
	os.remove(os.path.join(curdir, filename))
	print('removed', os.path.join(curdir, filename))
	if filename.endswith('.dat') or filename.endswith('.h') or filename.endswith('_state.txt'):
		os.system(f"ssh jtabet@login 'rm /scratch/data/jtabet/output/{filename}'")
	return count+1

directory = '/scratch.local/data/output'
directory_script = '/scratch.local/data/script_output'
directory_stationary='/scratch.local/data/output_stationary'
directory_stills = '/scratch.local/data/OwnCloud/Stills'
dirs = [directory, directory_script, directory_stationary, directory_stills]
count = 0
if len(sys.argv) > 1:
	all_argv = sys.argv[1:]
	for name in all_argv:
		sf = str(name)
		prefix = findpre(sf)
		for curdir in dirs:
			for filename in os.listdir(curdir):
				if filename.startswith(f'{prefix:s}{sf:s}_') and (filename.endswith('.npy') or filename.endswith('.dat') or filename.endswith('.h') or filename.endswith('.png')):
					count = remove_file(curdir, filename, count)					
				elif filename.startswith(f'{prefix:s}{sf:s}_') and filename.endswith('_state.txt'):
					if delete_state_files:
						count = remove_file(curdir, filename, count)
					else:
						os.system(f"ssh jtabet@login 'rm /scratch/data/jtabet/output/{filename}'")
	print(f'removed {count} files')
else:
	print('no arguments given\nenter any number of simulation names')