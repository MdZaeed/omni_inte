
import os
import pickle
from collections import namedtuple

import h5py as h5
import numpy as np
import pandas
from numpy import inf, nan
from scipy import stats

from util.UtilityClass import Utility
from collections import defaultdict


class RegionData:
    def __init__(self, data, runtime, eff_loss, variables):
        self.data = data
        self.runtime = runtime
        self.eff_loss = eff_loss
        self.variables = variables
        #print("Init with ", len(self.variables))

    def concat_to_data(self, new_data):
        if self.data is None:
            self.data = new_data
        else:
            self.data = np.hstack((self.data, new_data))
    
    def concat_to_runtime(self, new_runtime):
        if self.runtime is None:
            self.runtime = new_runtime
        else:
            self.runtime = np.vstack((self.runtime, new_runtime))
    
    def merge(self, new_reg_data):
        assert self.data is not None
        assert self.runtime is not None
        assert self.eff_loss is not None
        assert new_reg_data.data is not None
        assert new_reg_data.runtime is not None
        assert new_reg_data.eff_loss is not None

        if len(self.variables) != len(new_reg_data.variables):
            print("ERROR: Variables count does not match up")
            return None
        
        for i in range(len(self.variables)):
            assert self.variables[i] == new_reg_data.variables[i]

        new_data = np.concatenate((self.data, new_reg_data.data))
        new_runtime = np.concatenate((self.runtime, new_reg_data.runtime))
        new_eff_loss = np.concatenate((self.eff_loss, new_reg_data.eff_loss))

        assert new_data.shape[0] == new_runtime.shape[0]
        assert new_data.shape[1] == len(self.variables)

        return RegionData(new_data, new_runtime, new_eff_loss, self.variables)
    
    def flatten(self):
        self.data[self.data < 0] = np.nan
        self.runtime[self.runtime < 0] = np.nan

        #print(self.runtime)

        self.data = np.nanmean(self.data, axis=0)
        self.data = self.data.reshape(1, len(self.data))
        assert self.data.shape[0] == 1

        self.runtime = np.nanmean(self.runtime, axis=0)
        assert self.runtime.shape[0] == 1
    
    def copy(self):
        return RegionData(self.data, self.runtime, self.eff_loss, self.variables)





class DataLoader():

    @classmethod
    def create_instance(self, pkl_path=None, force_init=False, **kwargs):
        if os.path.isfile(pkl_path) and not force_init:
            print("Loading from %s..." % pkl_path)
            return DataLoader.load_state(pkl_path)
        else:
            data_loader = DataLoader(**kwargs)
            data_loader.save_state(pkl_path)
            return data_loader


    def __init__(self, h5_path=None, proc_nums=None, group_path=None,
            remove=False, norm=False, rescale=False):
        self.h5_path = h5_path
        self.proc_nums = proc_nums
        self.group_path = group_path
        self.remove = remove
        self.norm = norm
        self.rescale = rescale
        self.region_data_map = self.load_h5_data()
        self.region_data_full = self.merge_all()

        self.attributes, self.resources, self.resource_map, \
            self.ev_to_res_map, self.res_to_ev_map, self.uncore_flags = self.parse_resources()
        self.options = None
        #print(self.attributes)
        #print(len(self.attributes))
    
    def save_state(self, pkl_path):
        if not os.path.exists(os.path.dirname(pkl_path)):
            os.makedirs(os.path.dirname(pkl_path))
        
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(self, pkl_file)
    
    @classmethod
    def load_state(self, pkl_path):
        if not os.path.isfile(pkl_path):
            return None
        
        with open(pkl_path, 'rb') as pkl_file:
            return pickle.load(pkl_file)

    
    def update_options(self, new_options):
        if self.options:
            #print("Data_loader was loaded from a pkl, keeping original values...")
            for key in new_options:
                self.options[key] = new_options[key]
        else:
            self.options = new_options
    
    def get_option(self, option_name, default_val=None):
        if self.options and option_name in self.options:
            return self.options[option_name]
        else:
            return default_val
    
    def set_option(self, option_name, val):
        self.options[option_name] = val
  

    def load_h5_data(self):
        region_data_map = {}
        base_str = self.h5_path + 'perf-dump.%d_merged.h5'

        common_events_dict = defaultdict(int)
        for num_proc in self.proc_nums:
            h5_file_path = base_str % num_proc
            with h5.File(h5_file_path) as h5_file:
                for reg_key in h5_file.keys():
                    variables = list(h5_file[reg_key].keys())
                    variables.remove('Runtime')
                    for var in variables:
                        common_events_dict[var] += 1
        max_freq = max(common_events_dict.values())
        variables = []
        for ev_name in common_events_dict.keys():
            if (common_events_dict[ev_name] == max_freq):
                variables.append(ev_name)
            else:
                print("Removing: ", ev_name, common_events_dict[ev_name], max_freq)
        for num_proc in self.proc_nums:
            h5_file_path = base_str % num_proc
            region_data_map[num_proc] = {}

            print('Processing %s...' % h5_file_path)
            with h5.File(h5_file_path) as h5_file:
                for reg_key in h5_file.keys():
                    #print("Starting %s" % reg_key)
                    #variables = list(h5_file[reg_key].keys())
                    #variables.remove('Runtime')

                    #print(len(variables))
                    if reg_key not in region_data_map[num_proc]:
                        region_data_map[num_proc][reg_key] = RegionData(None, None, None, variables)

                    for ev_key in variables:
                        data = h5_file[reg_key][ev_key][:]
                        data = data[:, :, 0]
                        data = data.astype(float)
                        region_data_map[num_proc][reg_key].concat_to_data(data)

                    runtime = h5_file[reg_key]['Runtime'][:]
                    runtime = runtime[:, :, 0]
                    region_data_map[num_proc][reg_key].concat_to_runtime(runtime)

                    assert region_data_map[num_proc][reg_key].data.shape[0] == runtime.shape[0]
                    assert region_data_map[num_proc][reg_key].data.shape[1] == len(variables)

                    # Once the data's been verified, we flatten it by averaging
                    region_data_map[num_proc][reg_key].flatten()
        return region_data_map


    def merge_all(self):
        print("Attempting to merge all")
        reg_key_map = {}

        # We select the perf dump with the smallest number of processors
        # We extract its runtime across all of its region
        baseline_map = {}
        for proc_num in sorted(self.proc_nums):
            for reg_key in self.region_data_map[proc_num]:
                if reg_key not in baseline_map and self.region_data_map[proc_num][reg_key].runtime > 0:
                    baseline_map[reg_key] = self.region_data_map[proc_num][reg_key].runtime * proc_num
                    print("\nUsing proc=%d as a baseline for %s" % (proc_num, reg_key))
                    print("Baseline runtime for %s: %0.3f" % (reg_key, baseline_map[reg_key]))

        for num_proc in self.region_data_map:
            for reg_key in self.region_data_map[num_proc]:
                #print("Reg key : ", reg_key)
                # Set 'runtime' to be effeciency loss
                if reg_key in baseline_map:
                    baseline_runtime = baseline_map[reg_key]
                    app_runtime = self.region_data_map[num_proc][reg_key].runtime
                    #print(app_runtime)
                    self.region_data_map[num_proc][reg_key].eff_loss = 1 - (baseline_runtime / (num_proc * app_runtime))
                
                    if reg_key not in reg_key_map:
                        reg_key_map[reg_key] = []
                    reg_key_map[reg_key].append(self.region_data_map[num_proc][reg_key])
        
        merge_map = {}
        for reg_key in reg_key_map:
            for reg_data in reg_key_map[reg_key]:
                if reg_key not in merge_map:
                    merge_map[reg_key] = reg_data.copy()
                else:
                    new_merge = merge_map[reg_key].merge(reg_data)
                    if new_merge:
                        merge_map[reg_key] = new_merge
                    else:
                        pass
                        #print("Skipping...")
        
        print("Done merging")

        return merge_map

    def parse_resources(self):
        util = Utility()
        a_groups, uncore_flags, arch_dict = util.set_arch_groups()
        print (uncore_flags)
        event_list = util.get_event_list(self.group_path)
        event_groups = util.assign_event_list_to_eventGroups(event_list, a_groups)

        resources = [key for key in event_groups]
        attributes = [val for key in event_groups for val in event_groups[key]]
        attributes = list(set(attributes))

        resource_map = np.zeros((len(attributes), len(resources)), dtype=int)
        for j in range(len(resources)):
            resource_name = resources[j]

            for event in event_groups[resource_name]:
                i = attributes.index(event)
                resource_map[i, j] = 1
        
        assert np.size(resource_map,0) == len(attributes)
        assert np.size(resource_map,1) == len(resources)

        # this provide a event->resource lookup
        event_to_res_map = {}
        for res_key in event_groups:
            for ev_key in event_groups[res_key]:
                if ev_key not in event_to_res_map:
                    event_to_res_map[ev_key] = []
                event_to_res_map[ev_key].append(res_key)

        return attributes, resources, resource_map, event_to_res_map, event_groups, uncore_flags
    
    def set_region(self, reg_key):
        self.data = self.region_data_full[reg_key].data.copy()
        self.variables = self.region_data_full[reg_key].variables.copy()
        #print("Count: ", len(self.variables))
        self.eff_loss = self.region_data_full[reg_key].eff_loss
        self.runtime = self.region_data_full[reg_key].runtime
        self.event_map = self.generate_event_map()

        # Divides selected counters by their processor count
        divide_vec = np.array(self.proc_nums)
        for ev_index, event in enumerate(self.variables):
            res = self.ev_to_res_map[event][0] #Assuming a list of resources is returned
            if self.uncore_flags[res]:
                self.data[:, ev_index] /= divide_vec
            
        
        if self.norm:
            #print("Starting norm")
            with np.errstate(divide='ignore'):
                self.handle_norm()
            #print("Done with norm")

        if self.remove:
            #print("Starting remove")
            self.handle_remove()
            #print("Done with remove")
       
        if self.rescale:
            #print("Starting rescale")
            # We can divide by 0 here, but it's handled
            with np.errstate(divide='ignore', invalid='ignore'):
                self.data = self.handle_rescale(self.data)
            #print("Done with rescale")


    def get_full_app_data(self):
        return self.data.copy()

    def get_full_app_runtime(self):
        return self.runtime.copy().reshape(len(self.runtime))

    def get_eff_loss(self):
        return self.eff_loss.copy().reshape(len(self.eff_loss))

    
    def get_event_map(self):
        return self.event_map.copy()
    
    def get_event_indices(self, resource):
        indices = []
        for key in self.res_to_ev_map[resource]:
            if key in self.variables:
                indices.append(self.variables.index(key))
        
        return indices

    def generate_event_map(self):
        # TODO: update the info here
        # Create an event map of mxn where:
        # m is the number of variables we have
        # n is the number of operations(?) in event_to_resource_map
        m = len(self.variables)
        n = np.size(self.resource_map, 1)
        event_map = np.zeros((m, n))

        assert len(self.variables) == len(set(self.variables))
        assert len(self.attributes) == len(set(self.attributes))

        # Iterate over each variables and find where its corresponding
        # row in our event_to_resource map
        for i in range(m):
            relative_index = self.attributes.index(self.variables[i])
            event_map[i,:] = self.resource_map[relative_index,:]

        return event_map
    
    
    def handle_remove(self):
        # Removes any columns containing only zeroes
        empty_columns = np.where(np.sum(self.data, axis=0) == 0)[0]

        self.data = np.delete(self.data, empty_columns, axis=1)
        self.event_map = np.delete(self.event_map, empty_columns, axis=0)

        for index in sorted(empty_columns, reverse=True):
            #print(index)
            del self.variables[index]


    def handle_norm(self):
        # NOTE: This will throw errors when dividing by 0, this isn't preventable
        # Such entries are either set to NaN or +/- Inf depending on context
        np.seterr(all='ignore')
        self.data = stats.zscore(self.data, axis=0, ddof=1)
        np.seterr(all='print')

        # We clean up such entries here by setting them to zero
        self.data = self.clean_nan_and_inf(self.data)


    def handle_rescale(self, data):
        # Get the min of each row and tile it to the size of data
        min_array = np.tile(data.min(axis=0), (np.size(data,0), 1))
        # Get the max of each row (and minus the min) and tile it to the size of data
        max_array = np.tile(data.max(axis=0), (np.size(data,0), 1)) - min_array

        # Subtract the min such that the min of each row is now 0
        # Divide the (max-min) of each row such that the max of each row is now 1
        data -= min_array
        data = np.divide(data, max_array)
        data = self.clean_nan_and_inf(data)

        return data


    def clean_nan_and_inf(self, data):
        # Replace infinity and NaN with 0
        data[data == inf] = 0
        data[data == -inf] = 0
        data = np.nan_to_num(data)

        return data
    
    def get_regions(self):
        return list(self.region_data_full.keys())
    

    def get_rsm_scores(self, csv_path=None):
        if 'rsm_results' in self.options:
            return self.options['rsm_results']
        elif csv_path is None:
            print("Error: csv_path was not given and rsm scores were not in options")
        
        df = pandas.read_csv(csv_path)
        resources = df.columns[1:]
        regions = list(df.values[:, 0])
        data = df.values[:, 1:]
        data = data.astype(float)

        rsm_results = {}

        for reg_i, region in enumerate(regions):
            rsm_results[region] = {}
            for res_i, resource in enumerate(resources):
                rsm_results[region][resource] = data[reg_i, res_i]
        
        self.options['rsm_results'] = rsm_results
        return rsm_results

    def get_rsm_alphas(self, csv_path=None):
        if 'rsm_alphas' in self.options:
            return self.options['rsm_alphas']
        elif csv_path is None:
            print("Error: csv_path was not given and rsm scores were not in options")
        
        df = pandas.read_csv(csv_path)
        events = df.columns[1:]
        regions = list(df.values[:, 0])
        data = df.values[:, 1:]
        data = data.astype(float)

        rsm_alphas = {}

        for reg_i, region in enumerate(regions):
            rsm_alphas[region] = {}
            for ev_i, event in enumerate(events):
                rsm_alphas[region][event] = data[reg_i, ev_i]
        
        self.options['rsm_alphas'] = rsm_alphas
        return rsm_alphas


    def get_rsm_errors(self, csv_path=None):
        if 'rsm_errors' in self.options:
            return self.options['rsm_errors']
        elif csv_path is None:
            print("Error: csv_path was not given")
            return
            

        df = pandas.read_csv(csv_path)
        events = df.columns[1:]
        regions = list(df.values[:, 0])
        data = df.values[:, 1:]
        data = data.astype(float)

        rsm_alphas = {}

        for reg_i, region in enumerate(regions):
            rsm_alphas[region] = {}
            for ev_i, event in enumerate(events):
                rsm_alphas[region][event] = data[reg_i, ev_i]
        
        self.options['rsm_errors'] = rsm_alphas
        return rsm_alphas

    
    def get_rsm_belief(self, csv_path=None):
        if 'rsm_belief' in self.options:
            return self.options['rsm_belief']
        elif csv_path is None:
            print("Error: csv_path was not given and rsm_alphas were not in options")
            

        rsm_alphas = self.get_rsm_errors(csv_path=csv_path)
        rsm_belief = {}
        for region in rsm_alphas:
            rsm_belief[region] = {}
            for event in rsm_alphas[region]:
                val = rsm_alphas[region][event]
                belief = np.exp(-0.005 * val)
                rsm_belief[region][event] = belief
        
        self.options['rsm_belief'] = rsm_belief
        return rsm_belief
    
    def get_readable_reg_names(self):
        out = []

        for reg in self.get_regions():
            if ':' in reg:
                reg = reg.split(':')[-1]
            
            if '()' in reg:
                reg = reg.split('()')[0]
            
            out.append(reg)
        
        return out





if __name__ == "__main__":
    dl = DataLoader('data/nyx_data/', 'resources/native_all_filtered.txt',
        remove=False, rescale=False, norm=False)
    #print(dl.get_regions())
    
    key = dl.get_regions()[0]
    dl.set_region(key)
