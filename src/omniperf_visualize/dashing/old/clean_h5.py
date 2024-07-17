import h5py as h5
import sys
from shutil import copyfile
import os 
import numpy as np

def main(job_path, group_path, group_prefix):
	for file_name in os.listdir(job_path):
		dir_path = os.path.join(job_path, file_name)
		if os.path.isdir(dir_path):
			process_job(dir_path, group_path, file_name, group_prefix)

def process_job(dir_path, group_path, job_index, group_prefix):
	group_full_path = os.path.join(group_path, group_prefix + job_index)

	all_groups = []
	with open(group_full_path) as f:
		for line in f:
			events = line.split(',')
			new_events = []
			for event in events:
				new_events.append(event.rstrip())
			all_groups.append(new_events)

	for i in range(100):
		h5_file_path = os.path.join(dir_path, 'perf-dump.%d.h5' % i)
		new_hf_name = os.path.join(dir_path, 'perf-dump.%d_polished.h5' % i)

		if not os.path.isfile(h5_file_path):
			continue
	
		hf = h5.File(h5_file_path)

		print("Processing ", h5_file_path)

		if os.path.isfile(new_hf_name):
			os.remove(new_hf_name)
		copyfile(h5_file_path, new_hf_name)
		new_hf = h5.File(new_hf_name)

		assert len(hf.keys()) == len(new_hf.keys())

		for group_key in hf.keys():
			print(" - starting key: ", group_key)
			runtime = hf[group_key]["Runtime"]

			del new_hf[group_key]["Runtime"]

			for line_num, group in enumerate(all_groups):
				index = runtime.shape[-1] - 2 - line_num
				group_runtime = runtime[:,:,index]
				
				for j in range(group_runtime.shape[0]):
					process_runtime = group_runtime[j,0]
					if process_runtime < 0:
						for event in group:
							assert hf[group_key][event][j,0,0] == 0
							new_hf[group_key][event][j,0,0] = -1
			
			avg_runtime = np.zeros((i,1,2))
			for j in range(i):
				row_runtime = runtime[j,0]
				row_sum = 0.0
				row_count = 0
				for run in row_runtime:
					if run > 0:
						row_sum += run
						row_count += 1
		
				avg = -1
				if row_count > 0:
					avg = row_sum/row_count		

				avg_runtime[j][0][0] = avg
	
			new_hf[group_key]["Runtime"] = avg_runtime

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: " + sys.argv[0] + " path_to_jobs path_to_group_files group_prefix")
	else:
		main(sys.argv[1], sys.argv[2], sys.argv[3])

	#main()


