import os
from sys import exit
import h5py as h5
import numpy as np



def main():
	#check_event_existence_among_all()
	#check_region_existence_among_all()
	check_for_zeros()



def index_mapping(L):
	index_dict = {}
	for i in range(len(L)):
		index_dict[L[i]] = i
	return index_dict
	


def check_for_zeros():
	
	numProcs = 4
	base_str = 'perf-dump.%d_merged.h5'
	h5_files = [base_str % (num_threads) for num_threads in range(4,numProcs+1,4)]

	with h5.File(h5_files[-1]) as curr_h5: # assuming the highest thread count will have encountered everything
		regions = list(curr_h5.keys())
		events = list(curr_h5[regions[0]].keys())
		procs = [i for i in range(4, numProcs+1, 4)]

					#number of files, number of region names, number of events, number of processes
	del events[events.index("Runtime")]
	A = np.zeros((len(h5_files), len(regions), len(events), numProcs))
	A[:,:,:,:] = -1

	proc_map = index_mapping(procs)
	region_map = index_mapping(regions)
	event_map = index_mapping(events)

	for proc_count, proc_index in proc_map.items():
		with h5.File(h5_files[proc_index]) as hf:
			print("file: ", h5_files[proc_index])
			for reg_name, reg_index in region_map.items():
				print("region: ", reg_name)
				for eve_name, eve_index in event_map.items():
					#print(hf[reg_name][eve_name][1][0][0])
					for threadID in range(proc_count):
						A[proc_index, reg_index, eve_index, threadID] =\
							hf[reg_name][eve_name][threadID][0][0]
	
	for eve_name, eve_index in event_map.items():	
		x = np.argwhere(A[:,:,eve_index,:] > 0)
		count = x.shape[0]
		
		if count == 0:
			print("%s" % eve_name)

def check_event_existence_among_all():
	#do all of these files contain the same functions with the same events per function?
	#function -> list of events
	
	all_events = parseAllEvents("native.txt") # there are 270 native events

	base_str = 'perf-dump.%d_merged.h5'
	h5_files = [base_str % (num_threads) for num_threads in range(4,33,4)]
	
	for h5_file in h5_files:
		with h5.File(h5_file) as curr_h5:
			for reg_key in curr_h5.keys():

				#get all events for this region
				this_regions_events = []
				for event_key in curr_h5[reg_key].keys():
					if event_key not in this_regions_events:
						this_regions_events.append(event_key)
	
				#and see if there's any missing
				for event in all_events:
					if event not in this_regions_events:
						print(event, " missing from ", reg_key, " in ", h5_file)



def parseAllEvents(filename):
	with open(filename) as f:
		all_events = []
		for line in f:
			for event in line.split(','):
				all_events.append(event.rstrip())
	return all_events







def reference():
    for num_proc in range(4, 33, 4):
        base_str = 'jobs/%s/perf-dump.%d_polished.h5'
        int_to_str = lambda i : '0' + str(i) if i < 10 else str(i)
        h5_files = [base_str % (int_to_str(job_id), num_proc)
            for job_id in range(24)]
        
        print("\n-----------------------------------------")
        print("Starting merging of num_procs=%d" % num_proc)
        print("-----------------------------------------")
        merge_files(h5_files, num_proc)

def merge_files(h5_files, num_proc):
    merge_file = 'perf-dump.%d_merged.h5' % num_proc

    if os.path.isfile(merge_file):
        os.remove(merge_file)
    
    with h5.File(merge_file) as curr_h5:
        runtime_map = {}

        for h5_file in h5_files:
            with h5.File(h5_file) as h5_to_merge:
                for reg_key in h5_to_merge.keys():
                    # If this region was not seen yet, add it to our merge file
                    # and create a mapping in our runtime dict for later
                    if reg_key not in curr_h5.keys():
                        print("- Adding %s" % reg_key)
                        curr_h5.create_group(reg_key)
                        runtime_map[reg_key] = []
                    
                    # Add each event to the merge file
                    for event_key in h5_to_merge[reg_key].keys():
                        data = h5_to_merge[reg_key][event_key][:]

                        # Exception for runtime, we don't write its values yet
                        if event_key == 'Runtime':
                            runtime_map[reg_key].append(data)
                        else:
                            curr_h5[reg_key][event_key] = data
    
        # Now we begin processing our runtime
        for reg_key in runtime_map:
            runtime_sum = np.zeros((num_proc, 1, 2))
            count_arr = np.zeros((num_proc, 1, 2))
            count_arr[:, 0, 1] = 1  # last columns are just 0, set to 1 to avoid dividing by 0
            
            all_neg = True
            for runtime in runtime_map[reg_key]:
                for i in range(num_proc):
                    # We don't want to contribute to the average if -1
                    if runtime[i, 0, 0] > 0:
                        count_arr[i, 0, 0] += 1
                        runtime_sum[i, 0, 0] += runtime[i, 0, 0]
                        all_neg = False

            # If not a single runtime was positive, we simply set its runtime to -1
            # for all threads
            if all_neg:
                new_runtime = np.zeros((num_proc, 1, 2))
                new_runtime[:, 0, 0] = -1
                print("\nWARNING: %s had all negative runtime values\n" % reg_key)
            else:
                # Divide by count to average and then write out            
                new_runtime = np.divide(runtime_sum, count_arr)
                curr_h5[reg_key]['Runtime'] = new_runtime

            

if __name__ == "__main__":
    main()


#def uhafasdfiuohh():
#	for region in curr_h5.keys(): 
#				for event in curr_h5[region].keys():
#					for all_threads in curr_h5[region][event]:
#						for t in range(len(curr_h5[region][event])):
#							#print(h5_file, " | ", region, " | ", event, " | ", curr_h5[region][event][t][0][0])
#							pass
#							#break
#						break
#					break
#				break
#		break

