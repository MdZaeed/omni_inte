import numpy as np
import pandas
import warnings

from numpy import inf, nan
from scipy import stats
from typing import List, Optional

class DataLoader():
    base_app_path = None
    base_proxy_path = None
    resource_path = None
    workloads = None

    app_data = None
    app_runtime = None

    proxy_data = None
    proxy_runtime = None

    variables = None
    attributes = None
    resources = None
    resource_map = None
    event_map = None


    def __init__(self, base_app_path : str, base_proxy_path : str, resource_path : str,
            workloads : List[int], file_suffix : Optional[str]='',
            remove : Optional[bool]=False, norm : Optional[bool]=False,
            rescale : Optional[bool]=False) -> None:
        
        # Append each workload index to the end of the base paths
        app_path_files = self.gen_file_paths(base_app_path, workloads, file_suffix)
        proxy_path_files = self.gen_file_paths(base_proxy_path, workloads, file_suffix)
        
        # Iterate through each app file, parse and load, and stack each job on
        # top of eachother.
        # We may also pull out the variables here
        self.app_data, self.app_runtime, self.variables = \
            self.parse_and_stack_data(app_path_files)
    
        # Same thing with proxy data, except the variables are the same and we
        # don't need to note it again
        self.proxy_data, self.proxy_runtime, _ = \
            self.parse_and_stack_data(proxy_path_files)

        # Assert that our app data and proxy data are corresponding correctly
        assert np.size(self.app_data) == np.size(self.proxy_data)
        assert np.size(self.app_runtime) == np.size(self.proxy_runtime)

        # Given the path to the resource map, extracts the attributes, resources, and map
        self.attributes, self.resources, self.resource_map = \
            self.gen_resource_map(resource_path)

        # Using the resource map, we may now generate our event map
        self.event_map = self.generate_event_map()

        # If true, remove any empty columns from app_data and proxy_data
        # This directly alters app_data, proxy_data, variables, and event_map
        if remove:
            self.handle_remove()

        # If true, normalize the data
        if norm:
            self.handle_norm()
        else:
            self.app_data_full = self.app_data
            self.proxy_data_full = self.proxy_data
        
        # If true, rescale all values -> [0, 1]
        if rescale:
            self.app_data_full = self.handle_rescale(self.app_data_full)
            self.proxy_data_full = self.handle_rescale(self.proxy_data_full)
    

    def get_full_app_data(self):
        return self.app_data_full.copy()
    
    def get_full_proxy_data(self):
        return self.proxy_data_full.copy()
    
    def get_full_app_runtime(self):
        return self.app_runtime.copy()
    
    def get_event_map(self):
        return self.event_map.copy()


    def gen_file_paths(self, base_path, workloads, file_suffix):
        # Adds the workload index and file suffix to the end of base_path
        path_files = []
        for workload_index in workloads:
            suffix = str(workload_index) + file_suffix
            path_files.append(base_path + suffix)
        
        return path_files 


    def generate_event_map(self):
        # Create an event map of mxn where:
        # m is the number of variables we have
        # n is the number of operations(?) in event_to_resource_map
        m = len(self.variables)
        n = np.size(self.resource_map, 1)
        event_map = np.zeros((m, n))

        # Iterate over each variables and find where its corresponding
        # row in our event_to_resource map
        for i in range(m):
            relative_index = np.where(self.attributes==self.variables[i])[0]

            # Multiple indices shouldn't come up, but if they do...
            assert len(relative_index) == 1
            relative_index = relative_index[0]

            event_map[i,:] = self.resource_map[relative_index,:]

        return event_map


    def parse_and_stack_data(self, path_files):
        # Parses all the paths in path_files and stack the data on top of eachother
        stack_data = None
        stack_runtime = None
        variables = None
        for i, path in enumerate(path_files):
            # Grab the data, runtime, and variables
            data, runtime, variables = self.parse_data(path)

            # We initialize our data on our first itereration
            if stack_data is None:
                stack_data = data
                stack_runtime = runtime
            else:
                # Otherwise we stack the data
                stack_data = np.vstack((stack_data, data))
                stack_runtime = np.concatenate((stack_runtime, runtime))
            
            # TODO: I think vstack will throw an error anyways here if this fails
            # This is to assure that we are actually loading the data right
            assert (np.size(stack_data, 0) / np.size(data, 0)) == (i+1)
        
        return (stack_data, stack_runtime, variables)


    def gen_resource_map(self, resource_path):
        # Read the resource map as a csv
        pd = pandas.read_csv(resource_path, delimiter='\t')
        
        # Extract the values we care about
        resources = pd.columns.values[1:]
        attributes = pd.values[:,0]
        resource_map = pd.values[:,1:]

        # If resource_map is nxm, then:
        #   attributes = nx1
        #   resourcess = mx1
        # Assert that this is in fact true
        assert np.size(resource_map,0) == np.size(attributes)
        assert np.size(resource_map,1) == np.size(resources)

        return attributes, resources, resource_map


    def parse_data(self, path):
        # Simply parses the csv and extracts all the data we want
        pd = pandas.read_csv(path, delimiter='\t')

        data = pd.values[:,1:]  
        runtime = pd.values[:,0]
        variables = list(pd.columns)[1:]

        assert np.size(data,0) == np.size(runtime)
        assert np.size(data,1) == np.size(variables)

        return (data, runtime, variables)


    def handle_remove(self):
        # Removes any columns containing only zeroes
        empty_columns = np.where(np.sum(self.app_data, axis=0) == 0)[0]

        self.app_data = np.delete(self.app_data, empty_columns, axis=1)
        self.proxy_data = np.delete(self.proxy_data, empty_columns, axis=1)
        self.event_map = np.delete(self.event_map, empty_columns, axis=0)

        for index in sorted(empty_columns, reverse=True):
            del self.variables[index]


    def handle_norm(self):
        # NOTE: This will throw errors when dividing by 0, this isn't preventable
        # Such entries are either set to NaN or +/- Inf depending on context
        np.seterr(all='ignore')
        self.app_data_full = stats.zscore(self.app_data, axis=0, ddof=1)
        self.proxy_data_full = stats.zscore(self.proxy_data, axis=0, ddof=1)
        np.seterr(all='print')

        # We clean up such entries here by setting them to zero
        self.app_data_full = self.clean_nan_and_inf(self.app_data_full)
        self.proxy_data_full = self.clean_nan_and_inf(self.proxy_data_full)


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


    def writeout(self, path):
        # TODO: Save more properly, this is only here for validatin purposes
        np.savetxt(path + "python_app_data.csv", self.app_data, delimiter=',', fmt='%0.34f')
        np.savetxt(path + "python_app_data_full.csv", self.app_data_full, delimiter=',', fmt='%0.34f')
        np.savetxt(path + "python_proxy_data.csv", self.proxy_data, delimiter=',', fmt='%0.34f')
        np.savetxt(path + "python_proxy_data_full.csv", self.proxy_data_full, delimiter=',', fmt='%0.34f')
