import os
import sys
import h5py as h5
import numpy as np
from collections import defaultdict

def main(job_path, num_proc, dump_path):
    base_str = '%s/perf-dump.%s_polished.h5'
    base_path = os.path.join(job_path, base_str)
    int_to_str = lambda i : '0' + str(i) if i < 10 else str(i)
    
    h5_files = []
    for job_id in range(100):
        file_path = base_path % (int_to_str(job_id), num_proc)

        if os.path.isfile(file_path):
            h5_files.append(file_path)
    
    if len(h5_files) == 0:
        return

    print("\n-----------------------------------------")
    print("Starting merging of num_procs=%s" % num_proc)
    print("-----------------------------------------")
    merge_files(h5_files, int(num_proc), dump_path)
    #common_event_list = get_common_event_names_across_all_configs(h5_files, int(num_proc))
    
def get_common_event_names_across_all_configs(h5_files, num_proc):
    event_count = defaultdict(int)
    print (h5_files)
    for h5_file in h5_files:
        with h5.File(h5_file) as h5_to_merge:
            for reg_key in h5_to_merge.keys():
                for event_key in h5_to_merge[reg_key].keys():
                    event_count[event_key] += 1
                    print(event_key, reg_key, h5_file)
    common_event_list = []
    for ev_name, count in event_count.items():
        if (count < ):
            print (ev_name, count)
        else:
            common_event_list.append(ev_name)
    return common_event_list
                    
def merge_files(h5_files, num_proc, dump_path):
    merge_file = 'perf-dump.%d_merged.h5' % num_proc
    merge_file = os.path.join(dump_path, merge_file)

    if os.path.isfile(merge_file):
        os.remove(merge_file)
    
    with h5.File(merge_file) as merge_h5:
        runtime_map = {}

        for h5_file in h5_files:
            with h5.File(h5_file) as h5_to_merge:
                for reg_key in h5_to_merge.keys():
                    # If this region was not seen yet, add it to our merge file
                    # and create a mapping in our runtime dict for later
                    if reg_key not in merge_h5.keys():
                        print("- Adding %s" % reg_key)
                        merge_h5.create_group(reg_key)
                        runtime_map[reg_key] = []
                    
                    # Add each event to the merge file
                    for event_key in h5_to_merge[reg_key].keys():
                        data = h5_to_merge[reg_key][event_key][:]

                        # Exception for runtime, we don't write its values yet
                        if event_key == 'Runtime':
                            runtime_map[reg_key].append(data)
                        elif event_key not in merge_h5[reg_key].keys():
                            merge_h5[reg_key][event_key] = data
    
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
            
            merge_h5[reg_key]['Runtime'] = new_runtime

            

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: " + sys.argv[0] + " job_path proc_num dump_location")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
