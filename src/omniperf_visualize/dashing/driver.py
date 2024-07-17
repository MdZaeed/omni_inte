import yaml
import importlib
import numpy as np
from omniperf_visualize.dashing.util.pipeline import DataLoader
import os
import pickle
import sys
    
DRIVER_KEYS = set(['data', 'force_process', 'tasks', 'global_task'])

class driver(dict):
    def __init__(self):
        print(self)

    def remove_tmp_folder(self,glboal_config_filename, global_config_name):
        tmp_folder = os.path.join('data', 'tmp')
        file_prefix = '%s_%s_' % (glboal_config_filename, global_config_name)
        file_suffix = '.pkl'

        for file_name in os.listdir(tmp_folder):
            if file_name.startswith(file_prefix) and file_name.endswith(file_suffix):
                file_path = os.path.join(tmp_folder, file_name)
                print("Removing %s" % file_path)
                os.unlink(file_path)

    def main(self,config_file, force_compute, config_name=None):
        config_path, config_dict = self.load_config_file(config_file)

        if config_dict is None:
            return
        
        if config_name is None:
            config_name = 'main'
        
        config_filename = os.path.basename(config_path).rsplit('.', 1)[0]

        if config_name not in config_dict:
            print("ERROR: %s could not be found in %s. Please include this config or specify a valid config to find."
                % (config_name, config_path))
        else:
            # Forces a regeneration of all dataloaders
            if force_compute:
                print("\nForce Compute flag was detected, will recompute all Data Loaders.\n")
                # self.remove_tmp_folder(config_filename, config_name)

            self.handle_global_config(config_dict[config_name], config_name, config_dict, config_filename)
            
        
    def handle_global_config(self,config_dict, config_name, all_configs, config_filename):
        # These are options each driver_loader will get by default unless a config overrides this
        global_options = {}
        for key in config_dict:
            # We don't allow overriding of any "global" options
            if key not in DRIVER_KEYS:
                global_options[key] = config_dict[key]
            
        # Begin iterating over each task
        global_tasks = config_dict['tasks']    
        data_loaders = {}
        for task in global_tasks:
            print("===========================")
            print("Computing %s " % task)
            print("===========================")
            # if '.' not in task then we have a specific application's config to run
            if '.' not in task:
                # We let this config run and save the data_loader
                app_config = all_configs[task]
                app_config = self.update_dict(app_config, global_options)
                data_loaders[task] = self.run_simple_config(app_config, task, config_name, config_filename)
            else:
                # Otherwise we have a function to call
                module_name, func_name = task.rsplit('.', 1)
                module_name = 'omniperf_visualize.dashing.' + module_name
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                # We provide all data loaders and the config for this global task
                func(data_loaders, global_options)


    def update_dict(self,base_dict, update_dict):
        for key in update_dict:
            if key not in base_dict:
                base_dict[key] = update_dict[key]
            else:
                print("%s was overriden by a config." % key)
        return base_dict


    def run_simple_config(self,config_dict, config_name, global_config_name, global_config_filename):
        # A simple config is just a single applications tasks

        # data path is needed
        data_path = self.load_from_config(config_dict, 'data')
        procs = self.load_from_config(config_dict, 'procs', [], warning=False)
        tasks = self.load_from_config(config_dict, 'tasks', [])
        data_loader_path = self.load_from_config(config_dict,'data_loader')
        target = self.load_from_config(config_dict, 'target', ['Runtime'])
        compute_target = self.load_from_config(config_dict, 'compute_target')
        print(procs)
        #if data_path is None or procs is None: return
        if data_path is None: return

        #'arch_group_file', 'event_map_file', 'exclude_file', 'counters_file'])
        arch_name = self.load_from_config(config_dict, 'arch', 'haswell')
        temp_prefix = '/home/cup7/omni_inte/omniperf/src/omniperf_visualize/dashing/'
        arch_path = os.path.join(temp_prefix, 'resources', arch_name, 'architecture_groups.txt')
        event_path = os.path.join(temp_prefix, 'resources', arch_name, 'event_map.txt')
        exclude_path = os.path.join(temp_prefix, 'resources', arch_name, 'exclude_groups.txt')
        counters_path = os.path.join(temp_prefix, 'resources', arch_name, 'native_all_filtered.txt')
        desc_path = os.path.join(temp_prefix, 'resources', arch_name, 'event_desc.csv')

        # We make a dictionary to pass to the function for optional params
        options = {}
        for key in config_dict:
            if key not in DRIVER_KEYS:
                options[key] = config_dict[key]
        
        data_loader = self.load_state(global_config_filename, global_config_name, config_name)

        if not procs:
            procs = self.find_procs(data_path)
            procs = sorted(procs)
            print("Warning: procs was not found, using: ", procs)

        # Generate a data loader if we couldn't find a cached version
        if data_loader is None:
            if data_loader_path is None:
                file_ext = data_path.split('.')[-1]
                print(file_ext)
                if file_ext == 'csv':
                    data_loader = DataLoader.init_from_csv(config_name, data_path,
                        procs, arch_path, event_path, exclude_path, counters_path, target, compute_target)
                else:
                    data_loader = DataLoader.init_from_h5(config_name, data_path,
                        procs, arch_path, event_path, exclude_path, counters_path, target)
            # else:
            #     module_name, data_loader_func = task.rsplit('.', 1)
            #     print("Importing Data Loader from: %s" %func_name)
            #     module = importlib.import_module(module_name)
            #     func = getattr(module, func_name)
            #     data_loader = func(data_path)
            # self.save_state(global_config_filename, global_config_name, config_name, data_loader)

        # Update dataloader's internal dictionary
        data_loader.update_options(options)
        data_loader.options['charts'] = []
        data_loader['desc_path'] = desc_path
        # print(data_loader)
        # Run the tasks on this dataloader
        self.run_tasks(global_config_filename, global_config_name, config_name, data_loader, tasks)

        return data_loader


    def run_tasks(self,global_config_filename, global_config_name, config_name, data_loader, tasks):
        # We simply iterate over each function defined in our tasks,
        # load the module and function, and then run
        for task in tasks:
            module_name, func_name = task.rsplit('.', 1)
            print("Running %s..." % func_name)
            module_name = 'omniperf_visualize.dashing.' + module_name
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            if(data_loader is None) :
                print('YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
            result = func(data_loader)
            # self.save_state(global_config_filename, global_config_name, config_name, data_loader)

    def load_state(self,global_config_filename, global_config_name, config_name):
        pkl_path = os.path.join('data', 'tmp')
        pkl_path = os.path.join(pkl_path, '%s_%s_%s.pkl' \
            % (global_config_filename, global_config_name, config_name))

        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as pkl_file:
                print("Loading the state of %s-%s-%s..." \
                    % (global_config_filename, global_config_name, config_name))
                return pickle.load(pkl_file)
        return None

    def save_state(self,global_config_filename, global_config_name, config_name, data_loader):
        pkl_path = os.path.join('data', 'tmp')
        pkl_path = os.path.join(pkl_path, '%s_%s_%s.pkl' \
            % (global_config_filename, global_config_name, config_name))
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(data_loader, pkl_file)

    def load_config_file(self,config_path):
        if '.yml' not in config_path and '.yaml' not in config_path:
            config_path += '.yml'

        # if os.path.dirname(config_path) != 'configs':
        #     config_path = os.path.join('configs', config_path)
        
        if not os.path.isfile(config_path):
            print("ERROR: %s could be found. Please specify a valid config file." % config_path)
            return config_path, None

        with open(config_path, 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        return config_path, config_dict


    def load_from_config(self,config_dict, name, default_val=None, warning=True):
        if name not in config_dict:
            if default_val is not None:
                if warning:
                    print("Warning: '%s' was not defined, using default value of '%s'" \
                        % (name, default_val))
            else:
                print("Error: '%s' needs to be defined" % name)
            return default_val
        return config_dict[name]


    def find_procs(self,data_path):
        file_ext = data_path.split('.')[-1]
        print(file_ext)
        procs = set()

        if file_ext == 'csv':
            #These may not be directly related to process counts. These can be iterations x configurations x input. So keep procs_config, and num_configs[region] separate. And change accordingly. len(procs_config) may not be useful when num_configs[reg1] is not equal to num_configs[reg2]
            #For now, we are assuming that every region has the same number of configs.
            #proc_configs will be auto populated when we read the csv data in : parse_csv_data.
            print('Number of configs will be auto detected')
        else:
            data_path = os.path.join(data_path, "00")
            for filename in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, filename)) or "perf-dump" not in filename: continue

                proc_num = filename.split('.')[1][:2]
                if len(proc_num) > 1 and not proc_num[1].isdigit():
                    proc_num = proc_num[0]

                procs.add(int(proc_num))

        return list(procs)



    if __name__ == "__main__":
        usage_msg = "Usage: " + sys.argv[0] + " config_file <config_name> <-f>"


        if len(sys.argv) < 2:
            print(usage_msg)
            exit()
        
        args = sys.argv[1:]
        force_compute = False
        if '-f' in args:
            args.remove('-f')
            force_compute = True
        
        if len(args) == 1:
            main(args[0], force_compute)
        elif len(args) == 2:
            main(args[0], force_compute, config_name=args[1])
        else:
            print(usage_msg)
            exit()
