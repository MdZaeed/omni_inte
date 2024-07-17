import csv
import multiprocessing
import os

import numpy as np
from scipy.optimize import nnls
from scipy.linalg import lstsq
from numpy import NaN, inf
import psutil
import importlib
import pandas as pd
import sys
sys.path.append('modules')
from omniperf_visualize.dashing.modules import linearsieve
from omniperf_visualize.dashing.modules import vis_sieve as vs

COMPLETE_FLAG = 'rsm_complete_flag'

def compute_correlation_analysis(data_loader):
    # print("Zayed new_pd")
    new_pd = pd.DataFrame(data_loader.h5_map).T

    column_names_to_drop = data_loader.target
    new_pd.drop(column_names_to_drop, axis=1, inplace=True)
    #Drop the index column
    new_pd.reset_index(drop=True, inplace=True)

    data = []
    for column_name in new_pd.columns:
        dd = new_pd[column_name] #Get all rows (regions) for this column
        mean_val = []
        for val_per_region in dd: # For each region
            # print('Zayed val_per_region', val_per_region)
            # if isinstance(val_per_region, str):
                # row_list = val_per_region.split(',')
                # print(row_list)
                # tmp=[ float(m) for m in row_list ]
                mean_val.append(val_per_region.mean())
            # else:
            #     mean_val.append(val_per_region)
        data.append(mean_val)
    column_names= list(new_pd.columns)
    data=np.array(data)
    data=data.T
    print('Zayed data: ', data)
    # print(data)
    # print(data.shape)
    for i in range(5, 35, 5):
        s = linearsieve.Sieve(n_hidden=i, verbose=False).fit(data)
        x_latent=s.transform(data)
        # print(x_latent)

        ## Create the directory, if it does not exist
        if not os.path.exists("output/"):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("output")
        f = open('output/x_new_'+str(i)+'.txt', 'w')
        f.write(str(x_latent))
        vs.output_groups_new(s.tcs, s.mis, column_names, 'new' , num_hidden_layers=i)

    print(new_pd.head())
    

def compute_rsm_task_all_regions(data_loader):
    if data_loader.get_option(COMPLETE_FLAG, False):
        print("Skipping RSM as it was saved in data_loader...")
        return

    num_iters = data_loader.get_option('rsm_iters', 2500)
    print_results = data_loader.get_option('rsm_print', False)
    use_nn_solver = data_loader.get_option('rsm_use_nn_solver', False)
    num_cpus = data_loader.get_option('rsm_cpu_count', None)
    rescale = data_loader.get_option('data_rescale', True)

    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False)
        print("WARNING: 'rsm_cpu_count' was not set, using %d as default" % num_cpus)

    # csv files to dump
    csv_rsm_results = data_loader.get_option('csv_rsm_results', None)
    csv_rsm_res_errors = data_loader.get_option('csv_rsm_res_errors', None)
    csv_rsm_ev_errors = data_loader.get_option('csv_rsm_ev_errors', None)
    csv_rsm_alphas = data_loader.get_option('csv_rsm_alphas', None)

    if use_nn_solver:
        print("Will use nn solver for RSM")
    else:
        print("Will use lstsq solver for RSM")

    results = {}
    errors = {}
    ev_errors = {}
    alphas = {}
    norm_data = {}
    for key in data_loader.get_regions():
        print("\n--------------")
        print("Region: ", key)

        rsm_score, error, ev_error, alpha, norm_d = compute_rsm(data_loader, key,
            num_iters=num_iters, use_nn_solver=use_nn_solver, num_cpus=num_cpus, rescale=rescale)
        
        rsm_dict = {}
        err_dict = {}
        for i in range(len(rsm_score)):
            if np.isnan(rsm_score[i]):
                rsm_score[i] = 0
            rsm_dict[data_loader.resources[i]] = rsm_score[i]
            err_dict[data_loader.resources[i]] = error[i]

        alpha_dict = {}
        ev_err_dict = {}
        for i in range(len(data_loader.events)):
            alpha_dict[data_loader.events[i]] = alpha[i]
            ev_err_dict[data_loader.events[i]] = ev_error[i]


        results[key] = rsm_dict
        alphas[key] = alpha_dict
        errors[key] = err_dict
        ev_errors[key] = ev_err_dict
        norm_data[key] = norm_d
    
    data_loader['rsm_results'] = results
    data_loader['rsm_res_errors'] = errors
    data_loader['rsm_ev_errors'] = ev_errors
    data_loader['rsm_alphas'] = alphas
    data_loader['rsm_norm_data'] = norm_data
    data_loader[COMPLETE_FLAG] = True

    if print_results:
        for reg_key in results:
            print("\n%s\n---------------------" % reg_key)
            for resource in results[reg_key]:
                print("%s = %s" % (resource, results[reg_key][resource]))


    if csv_rsm_results:
        dump_rsm_data(results, csv_rsm_results)
    
    if csv_rsm_res_errors:
        dump_rsm_data(errors, csv_rsm_res_errors)
    
    if csv_rsm_ev_errors:
        dump_rsm_data(ev_errors, csv_rsm_ev_errors)
    
    if csv_rsm_alphas:
        dump_rsm_data(alphas, csv_rsm_alphas)


def dump_rsm_data(data_dict, csv_path):
    header = [val for key in data_dict for val in data_dict[key]]
    header = list(set(header))
    header.insert(0, 'Region')

    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for region in data_dict:
            row = [region]
            for val in header[1:]:
                row.append(data_dict[region][val])

            csv_writer.writerow(row)



def compute_rsm(data_loader, region, num_iters=2500, use_nn_solver=False,
        num_cpus=None, rescale=False):
    app_data = data_loader.get_app_data(region, rescale=rescale)
    #eff_loss = data_loader.get_app_eff_loss(region)
    #### TODO: Change the name from "eff_loss" to signal or target to generalize
    #eff_loss = data_loader.    
    module_name, func_name = data_loader.compute_target.rsplit('.', 1)
    module_name = 'omniperf_visualize.dashing.' + module_name
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    # We provide all data loaders and the config for this global task
    eff_loss = func(data_loader, region)


    # Equivalent to: A = A * diag(1 ./ sqrt(sum(A^2)))
    # We ignore divide by 0 since we repair afterwards
    with np.errstate(divide='ignore'):
        app_data = np.dot(app_data, np.diag(
            np.divide(1,np.sqrt(np.sum(np.square(app_data), axis=0)))))

    # We may be dividing by zero if an entire column is 0
    app_data[app_data == inf] = 0
    app_data[app_data == -inf] = 0
    app_data = np.nan_to_num(app_data)


    raw_alpha = ensemble_omp_wrapper(app_data, eff_loss, num_iters=num_iters,
        use_nn_solver=use_nn_solver, num_cpus=num_cpus)
    alpha = np.mean(raw_alpha, axis=1)  # m x 1

    lam = 0.005
    errors = np.zeros(len(data_loader.resources))
    event_set = set(data_loader.get_events())
    for i, resource in enumerate(data_loader.resources):
        resource_events = data_loader.get_events_from_resource(resource)
        ids = [data_loader.events.index(ev) for ev in resource_events if ev in event_set]
        #print(resource)
        #print(app_data[:,ids])
        # If this group has no events associated with it
        # We mark it's error as nan
        if len(ids) == 0:
            errors[i] = NaN
        else:
            # Belief propagation
            errors[i] = np.linalg.norm(
                eff_loss - np.dot(app_data[:,ids], alpha[ids]))
    
    ev_errors = np.zeros(len(data_loader.events))
    for i, event in enumerate(data_loader.events):
        ev_errors[i] = np.linalg.norm(
            eff_loss - np.dot(app_data[:,i], alpha[i]))


    base_error = np.linalg.norm(eff_loss)
    rsm_score = base_error - errors
    rsm_score /= base_error

    rsm_score[rsm_score < 0] = 0.0

    #m = np.exp(-lam * errors)
    #m = (m - min(m)) / (max(m) - min(m))
    #rsm_score = m.T

    return rsm_score, errors, ev_errors, alpha, app_data


def compute_rsm_with_data(data_loader, app_data, eff_loss, num_iters=5000, use_nn_solver=True,
        num_cpus=None, rescale=False):
    
    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False)
        print("WARNING: 'rsm_cpu_count' was not set, using %d as default" % num_cpus)

    # Equivalent to: A = A * diag(1 ./ sqrt(sum(A^2)))
    # We ignore divide by 0 since we repair afterwards
    with np.errstate(divide='ignore'):
        app_data = np.dot(app_data, np.diag(
            np.divide(1,np.sqrt(np.sum(np.square(app_data), axis=0)))))

    # We may be dividing by zero if an entire column is 0
    app_data[app_data == inf] = 0
    app_data[app_data == -inf] = 0
    app_data = np.nan_to_num(app_data)

    raw_alpha = ensemble_omp_wrapper(app_data, eff_loss, num_iters=num_iters,
        use_nn_solver=use_nn_solver, num_cpus=num_cpus)
    alpha = np.mean(raw_alpha, axis=1)  # m x 1

    lam = 0.005
    errors = np.zeros(len(data_loader.resources))
    event_set = set(data_loader.get_events())
    # print(data_loader.resources)
    for i, resource in enumerate(data_loader.resources):
        resource_events = data_loader.get_events_from_resource(resource)
        ids = [data_loader.events.index(ev) for ev in resource_events if ev in event_set]
        print(ids)
        #print(resource)
        #print(app_data[:,ids])
        # If this group has no events associated with it
        # We mark it's error as nan
        if len(ids) == 0:
            errors[i] = NaN
        else:
            # Belief propagation
            errors[i] = np.linalg.norm(
                eff_loss - np.dot(app_data[:,ids], alpha[ids]))
    
    ev_errors = np.zeros(len(data_loader.events))
    for i, event in enumerate(data_loader.events):
        ev_errors[i] = np.linalg.norm(
            eff_loss - np.dot(app_data[:,i], alpha[i]))


    base_error = np.linalg.norm(eff_loss)
    rsm_score = base_error - errors
    rsm_score /= base_error

    rsm_score[rsm_score < 0] = 0.0

    #m = np.exp(-lam * errors)
    #m = (m - min(m)) / (max(m) - min(m))
    #rsm_score = m.T

    return rsm_score, errors, ev_errors, alpha, app_data


def ensemble_omp_wrapper(D, Y, SPARSITY=15, THRESH=3, num_iters=2500,
        use_nn_solver=False, num_cpus=None):
    
    if num_iters % num_cpus == 0:
        step = num_iters // num_cpus
    else:
        step = (num_iters + num_cpus) // num_cpus

    tasks = []
    for _ in range(num_cpus):
        if num_iters > step:
            num_iters -= step
            proc_iters = step
        else:
            proc_iters = num_iters

        tasks.append((D.copy(), Y.copy(), SPARSITY, THRESH, proc_iters, use_nn_solver))

    print("Starting ensemble_omp...")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(ensemble_omp, tasks)
    print("Finished ensemble_omp!")

    alpha = results[0]
    for i in range(1, num_cpus):
        alpha = np.hstack((alpha, results[i]))

    return alpha


def ensemble_omp(ensemble_args):
    D, Y, SPARSITY, THRESH, num_iters, use_nn_solver = ensemble_args

    # D is app data, X is app_runtime
    K = D.shape[1]
    alpha = None
    for iter in range(num_iters):

        if (iter+1) % 500 == 0:
            print("Starting iteration %d/%d" % (iter+1, num_iters))

        A = []
        residual = Y.copy()
        index = np.zeros(SPARSITY, dtype=int)
        valid_indices = 0
        for i in range(SPARSITY):
            proj = np.abs(np.dot(D.T, residual))
            ids = np.argsort(proj)[::-1]

            proj[ids[THRESH:]] = 0
            proj = proj / np.sum(proj)

            pos = discrete_sample(proj)
            index[i] = pos
            valid_indices += 1

            A = D[:, index[:i+1]]
            
            if use_nn_solver:
                w = nnls(A, Y)[0]
            else:
                w = lstsq(A, Y)[0]

            residual = Y - np.dot(D[:, index[:i+1]], w)

            if np.isclose(np.sum(residual), 0.0):
                break

        B = np.zeros(K)
        B[index[:valid_indices]] = w
        
        B = np.reshape(B, (-1, 1))

        if alpha is None:
            alpha = B
        else:
            alpha = np.hstack((alpha, B))

    return alpha

def discrete_sample(probs):
    edges = np.append([0], np.cumsum(probs))
    s = edges[-1]
    if abs(s - 1) > np.finfo(float).eps:
        edges = edges * (1 / s)
    
    rv = np.random.random(1)
    bin_placement = np.digitize(rv, edges)

    if bin_placement == len(edges):
        bin_placement -= 2
    else:
        bin_placement -= 1

    return bin_placement
