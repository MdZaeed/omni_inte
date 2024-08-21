##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

from omniperf_visualize.visualize_base import OmniVisualize_Base
from utils.utils import demarcate, console_error
from utils import file_io, parser, tty
from utils.kernel_name_shortener import kernel_name_shortener
import pandas as pd
import os, shutil
import pathlib
import math
import copy
from collections import defaultdict
from omniperf_visualize.dashing.driver import driver
from omniperf_base import Omniperf
import numpy as np
from omniperf_visualize.dashing_setup import dashing_setup

LDS_HITS_COL = 'SQ_LDS_IDX_ACTIVE'
LDS_MISS_COL = 'SQ_LDS_BANK_CONFLICT'
L1D_COL = 'TCP_TOTAL_CACHE_ACCESSES_sum'
L2_WRITE_COL = 'TCP_TCC_WRITE_REQ_sum'
L2_ATOMIC_WRET_REQ_COL = 'TCP_TCC_ATOMIC_WITH_RET_REQ_sum'
L2_ATOMIC_WORET_REQ_COL = 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'
L2_READ_COL = 'TCP_TCC_READ_REQ_sum'

# DANIEL: ADDING NEW VARIABLES FROM OMNIPERF
HBM_READ_STALL_COL = 'TCC_EA_RDREQ_DRAM_CREDIT_STALL_sum'
HBM_WRITE_STALL_COL = 'TCC_EA_WRREQ_DRAM_CREDIT_STALL_sum'
HBM_READ_COL = 'TCC_EA_RDREQ_DRAM_sum'
HBM_WRITE_COL = 'TCC_EA_WRREQ_DRAM_sum'
VALU_INSTS_COL = 'SQ_INSTS_VALU'
MFMA_INSTS_COL = 'SQ_INSTS_MFMA'

VALU_FACTOR = 64
MFMA_FACTOR = 512

BYTES_PER_CACHE_LINE = 64
NUM_L2_BANKS = 32
LDS_MAGIC_NUMBER = 4

class visualize_cli(OmniVisualize_Base):

    def calculate_bw(self, df, bytes_col, output_col):

        # df['attainable_' + bytes_col] = df[['peak_valu_flops','intensity']].values.min(axis=1)
        # df['attainable_kernel_opsec'] = min(df['curr_ai'] * df['HBM_bytes'], peak_valu_flops)
        df[output_col] = df['total_flops'] / df[bytes_col]

        return df['temp']
    
    def parse_roofline_data(self,roof_df):
        roofline_data_map = {}

        # the device id in roofline.csv (0, 1) does not match the gpu ids in pmc_perf.csv (2, 3)
        # so use two separate counters
        roofline_counter = 0
        peak_bw_lds = roof_df['LDSBw'][roofline_counter]
        peak_bw_l1d = roof_df['L1Bw'][roofline_counter]
        peak_bw_l2 = roof_df['L2Bw'][roofline_counter]
        peak_bw_hbm = roof_df['HBMBw'][roofline_counter]
            # These two values are the peak valu and mfma performance for a certain data type.
            # They need to be changed if you are using a different data type.
            # For VALU, The options are: FP64Flops and FP32Flops
            # For MFMA, the options are: MFMAF64Flops, MFMAF32Flops, MFMAF16Flops, MFMABF16Flops
        peak_valu_flops = roof_df['FP32Flops'][roofline_counter]
        peak_mfma_flops = roof_df['MFMAF32Flops'][roofline_counter]

        roofline_data_map = (peak_bw_lds, peak_bw_l1d, peak_bw_l2, peak_bw_hbm, peak_valu_flops, peak_mfma_flops)

        roofline_counter += 1
        
        return roofline_data_map    

    def find_lds_bytes(self,df_row):
        return NUM_L2_BANKS * LDS_MAGIC_NUMBER * \
            (df_row[LDS_HITS_COL] - df_row[LDS_MISS_COL])

    def find_l1d_bytes(self,df_row):
        return BYTES_PER_CACHE_LINE * df_row[L1D_COL]

    def find_l2_bytes(self,df_row):
        return BYTES_PER_CACHE_LINE * \
                (df_row[L2_WRITE_COL] + df_row[L2_WRITE_COL] + \
                df_row[L2_ATOMIC_WRET_REQ_COL] + df_row[L2_ATOMIC_WORET_REQ_COL] )

    def find_hbm_bytes(self,df):
        # bytes of HBM traffic is a bit confusing to parse at first glance
        # but traffic is split between 32B and 64B traffic. reads keep track of 32B reads
        # and all reads, and writes keep track of 64B writes and all writes
        return  (
                (df["TCC_EA_RDREQ_32B_sum"] * 32)
                + ((df["TCC_EA_RDREQ_sum"] - df["TCC_EA_RDREQ_32B_sum"]) * 64)
                + (df["TCC_EA_WRREQ_64B_sum"] * 64)
                + ((df["TCC_EA_WRREQ_sum"] - df["TCC_EA_WRREQ_64B_sum"]) * 32)
                )

    def find_valu_ops_bytes_mi200(self,df_row):
        valu_fops = VALU_FACTOR * \
                    (df_row["SQ_INSTS_VALU_ADD_F16"] + \
                    df_row["SQ_INSTS_VALU_MUL_F16"] + \
                    2 * df_row["SQ_INSTS_VALU_FMA_F16"] + \
                    df_row["SQ_INSTS_VALU_TRANS_F16"]) + \
                    VALU_FACTOR * \
                    (df_row["SQ_INSTS_VALU_ADD_F32"] + \
                    df_row["SQ_INSTS_VALU_MUL_F32"] + \
                    2 * df_row["SQ_INSTS_VALU_FMA_F32"] + \
                    df_row["SQ_INSTS_VALU_TRANS_F32"]) + \
                    VALU_FACTOR * \
                    (df_row["SQ_INSTS_VALU_ADD_F64"] + \
                    df_row["SQ_INSTS_VALU_MUL_F64"] + \
                    2 * df_row["SQ_INSTS_VALU_FMA_F64"] + \
                    df_row["SQ_INSTS_VALU_TRANS_F64"])

        return valu_fops
        # This line is currently commented out since we do not track INT32/INT64 at this time.
        # Could be tracked in the future.
        ##valu_iops = VALU_FACTOR * (df_row['SQ_INSTS_VALU_INT32'] + df_row['SQ_INSTS_VALU_INT64'])
        ##total_valu_ops = valu_fops + valu_iops

    def find_mfma_ops_bytes_mi200(self,df_row):
        mfma_fops = MFMA_FACTOR * (df_row["SQ_INSTS_VALU_MFMA_MOPS_F16"] + df_row["SQ_INSTS_VALU_MFMA_MOPS_BF16"] + \
                                df_row["SQ_INSTS_VALU_MFMA_MOPS_F32"] + df_row["SQ_INSTS_VALU_MFMA_MOPS_F64"])

        return mfma_fops

    def generate_dash_data_format(self, df, feature_list, target, regions, outfilename, region_column="app"):
        # feature_list_and_target = feature_list[:]
        feature_list_and_target = feature_list[:]
        feature_list_and_target.append(target)
        
        with open(outfilename, 'w') as txtfile:   
            s = ','
            s += ','.join([str(item) for item in regions])
            s += '\n'
            txtfile.write(s)

            for feat in feature_list_and_target:
                if feat == region_column:
                    continue
                row=feat# + ",\""
                for reg in regions:
                    row += ",\""
                    ## GEt all rows that have the same kernel/region name
                    df_tmp = df[(df[region_column] == reg )] # DEBUG for empty values              
                    vals = []
                    
                    ## Now, make their values into a list
                    [vals.append(v) for v in df_tmp[feat].values]
                    s = ','.join([str(v) for v in vals])
                    row += s + "\""
                row += '\n'
                txtfile.write(row)  

    def find_common_features_and_regions(self, df_dict_param, filter_list, region_column="app"):
        # df_dist: list of workloads within a model
        # filter_list: columns (metrics/counters) we don't want

        features_dict = defaultdict(int)
        common_region_dict = defaultdict(int)
        regions_list_with_dup = [] # "region" = kernel
        feature_list = [] # "feature" = column (metric/counter)
        
        #For each file
        for key, df_instance in df_dict_param.items():
            #read in the file
            df_original2 = df_instance.copy()
            ## REMOVE all columns that contain only 0s
            df_original2 = df_original2.loc[:, (df_original2 != 0).any(axis=0)]
            for v in df_original2.columns:
                features_dict[v] += 1
                
            for v in df_original2[region_column]:
                common_region_dict[v] += 1
                #regions_list_with_dup.append(v)
                
        ## Finding the unique names of regions since regions_list_with_dup may have duplicate region names.
        ##Here, region == kernel
    #     region_list = list(set(regions_list_with_dup))

        ## FINDING COMMON FEATURES
        common_feature_list = []
        for k, v in features_dict.items():
            if k in filter_list:
                continue
            if v == len(df_dict_param):
                common_feature_list.append(k)

                
        ## FINDING COMMON Regions
        common_region_list = []
        for k, v in common_region_dict.items():
            if v == len(df_dict_param):
                common_region_list.append(k)

            
        return common_feature_list, common_region_list

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        if self.get_args().random_port:
            console_error("--gui flag is required to enable --random-port")
        for d in self.get_args().path:
            file_io.create_df_kernel_top_stats(
                raw_data_dir=d[0],
                filter_gpu_ids=self._runs[d[0]].filter_gpu_ids,
                filter_dispatch_ids=self._runs[d[0]].filter_dispatch_ids,
                time_unit=self.get_args().time_unit,
                max_stat_num=self.get_args().max_stat_num,
                kernel_verbose=self.get_args().kernel_verbose,
            )
            # create 'mega dataframe'
            self._runs[d[0]].raw_pmc = file_io.create_df_pmc(
                d[0], self.get_args().kernel_verbose, self.get_args().verbose
            )
            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(
                self._runs[d[0]].raw_pmc, self.get_args().kernel_verbose
            )

            # create the loaded table
            parser.load_table_data(
                workload=self._runs[d[0]],
                dir=d[0],
                is_gui=False,
                debug=self.get_args().debug,
                verbose=self.get_args().verbose,
            )

    @demarcate
    def run_visualize(self):
        """Run importance analysis visualization."""
        super().run_visualize()
        print('First run of visualize Zaeed')
        parent_folder_path = str(self.get_args().path[0][0])
        user_target = str(self.get_args().target).split(',')
        port = str(self.get_args().port)
        inverse = str(self.get_args().inverse)

        print(parent_folder_path,user_target,port,inverse)
        dashing_set = dashing_setup()
        dashing_set.run_visualize(parent_folder_path,user_target, port, inverse)

        # folder_names = [name for name in os.listdir(parent_folder_path) 
        #                 if os.path.isdir(parent_folder_path+'/'+name) and name!='temp']
        #         # folder_names = [name for name in os.listdir(str(self.get_args().path[0][0])) if os.path.isdir(name)]
        # print(folder_names)

        # file_names = ['pmc_perf.csv', 'SQ_IFETCH_LEVEL.csv', 'SQ_INST_LEVEL_LDS.csv', 'SQ_INST_LEVEL_SMEM.csv',
        #       'SQ_INST_LEVEL_VMEM.csv', 'SQ_LEVEL_WAVES.csv']
        # filter_list = ['Index', 'GPU_ID', 'queue-id', 'queue-index', 'pid', 'tid', 
        #        'grd', 'wgr', 'lds', 'scr', 'arch_vgpr', 'accum_vgpr', 'sgpr', 'wave_size', 'sig' 
        #        , 'obj', 'DispatchNs', 'BeginNs', 'EndNs', 'CompleteNs', 'obj_1', 'obj_2', 'obj_3',
        #        'obj_4', 'obj_5', 'obj_6', 'obj_7', 'obj_8', 'obj_9', 'obj_10', 'obj_11', 'obj_12',
        #        'obj_13', 'obj_14', 'obj_15', 'obj_16', 'obj_17', 'obj_18', 'Grid_Size', 'Workgroup_Size']
        # filter_list_plus = filter_list[:]
        # filter_list_plus.append('Kernel_Name')

        # temp_folder_paths = ['/home/mohammad/omni_inte/build/temp','/home/mohammad/omni_inte/build/temp_2']
        # for temp_folder_path in temp_folder_paths:
        #     for filename in os.listdir(temp_folder_path):
        #         file_path = os.path.join(temp_folder_path, filename)
        #         try:
        #             if os.path.isfile(file_path) or os.path.islink(file_path):
        #                 os.unlink(file_path)
        #             elif os.path.isdir(file_path):
        #                 shutil.rmtree(file_path)
        #         except Exception as e:
        #             print('Failed to delete %s. Reason: %s' % (file_path, e))

        # # return

        # pathlib.Path("./temp/").mkdir(parents=True, exist_ok=True)

        # # individual_run_dfs = {}
        # # for folder_name in folder_names:
        # #     prefix = parent_folder_path + '/' + folder_name + '/MI200/' 
        # #     df = pd.read_csv(prefix + 'timestamps.csv')
        # #     df['runtime'] = df['End_Timestamp'] - df ['Start_Timestamp']
        # #     temp_filter_list = list(set(filter_list) & set(df.columns))
        # #     df = df.drop(columns=temp_filter_list)
        # #     for file_name in file_names:
        # #         temp_df = pd.read_csv(prefix + file_name)
        # #         temp_filter_list = list(set(filter_list_plus) & set(temp_df.columns))
        # #         # print(temp_filter_list)
        # #         temp_df = temp_df.drop(columns=temp_filter_list)
        # #         # print(temp_df.shape)
        # #         df = pd.concat([df, temp_df], axis=1)
        # #     df = df.drop(columns=['Dispatch_ID'])
        # #     # print(df.columns)
        # #     individual_run_dfs[folder_name] = df
        # #     df.to_csv("./temp/" +  folder_name + '.csv')

        # individual_run_dfs = {}
        # for folder_name in folder_names:
        #     prefix = parent_folder_path + '/' + folder_name + '/MI200/' 
        #     df = pd.read_csv(prefix + 'timestamps.csv')
        #     df['runtime'] = df['End_Timestamp'] - df ['Start_Timestamp']
        #     temp_filter_list = list(set(filter_list) & set(df.columns))
        #     df = df.drop(columns=temp_filter_list)
        #     for file_name in file_names:
        #         temp_df = pd.read_csv(prefix + file_name)
        #         temp_filter_list = list(set(filter_list_plus) & set(temp_df.columns))
        #         temp_remove_list = list(set(df.columns) & set(temp_df.columns))
        #         # combined_filter_list = temp_filter_list
        #         combined_filter_list = list(set(temp_filter_list + temp_remove_list))
        #         if 'Dispatch_ID' in combined_filter_list:
        #             combined_filter_list.remove('Dispatch_ID')
        #         # print(temp_filter_list)
        #         # print(temp_remove_list)
        #         # print("WHYYYYYYYYYYYYYYYYYYYYYY")
        #         # print(combined_filter_list)
        #         temp_df = temp_df.drop(columns=combined_filter_list)
        #         # temp_df = temp_df.drop(columns=temp_filter_list)
        #         # print(temp_df.shape)
        #         # df = pd.concat([df, temp_df], axis=1)
        #         df = df.join(temp_df.set_index('Dispatch_ID'), on='Dispatch_ID')
        #     df = df.drop(columns=['Dispatch_ID'])
        #     # print(df.columns)
        #     individual_run_dfs[folder_name] = df
        #     df.to_csv("./temp/" +  folder_name + '.csv')

        # extended_individual_dfs = {}
        # for folder_name in folder_names:

        #     roofline_df = pd.read_csv(parent_folder_path + '/' + folder_name + '/MI200/' + 'roofline.csv')
        #     roofline_max_map = self.parse_roofline_data(roofline_df)
        #     peak_bw_lds, peak_bw_l1d, peak_bw_l2, peak_bw_hbm, peak_valu_flops, peak_mfma_flops = roofline_max_map
        #     # print(peak_bw_lds)

        #     new_df = individual_run_dfs[folder_name]
        #     # new_df = pd.read_csv("./temp/" +  folder_name + '.csv')
        #     new_df['VALU_ops'] = self.find_valu_ops_bytes_mi200(new_df)
        #     new_df['MFMA_ops'] = self.find_mfma_ops_bytes_mi200(new_df)
        #     new_df['LDS_bytes'] = self.find_lds_bytes(new_df)
        #     new_df['L1D_bytes'] = self.find_l1d_bytes(new_df)
        #     new_df['L2D_bytes'] = self.find_l2_bytes(new_df)
        #     new_df['HBM_bytes'] = self.find_hbm_bytes(new_df)

        #     new_df['total_flops'] = new_df['VALU_ops'] + new_df['MFMA_ops']
        #     new_df['OI_HBM'] = np.where(new_df['HBM_bytes'] == 0, 0, new_df['total_flops'] / new_df['HBM_bytes'])
        #     new_df['OI_LDS'] = np.where(new_df['LDS_bytes'] == 0, 0, new_df['total_flops'] / new_df['LDS_bytes'])
        #     new_df['OI_L1D'] = np.where(new_df['L1D_bytes'] == 0 , 0, new_df['total_flops'] / new_df['L1D_bytes'])
        #     new_df['OI_L2D'] = np.where(new_df['L2D_bytes'] == 0 , 0, new_df['total_flops'] / new_df['L2D_bytes'])
            
        #     # new_df['curr_ai'] = new_df['total_flops'] / new_df['HBM_bytes']
        #     # new_df['kernel_opsec'] = new_df['total_flops'] / new_df['runtime']

        #     # new_df['peak_valu_flops'] = peak_valu_flops
        #     # # new_df['intensity'] = new_df['curr_ai'] * new_df['HBM_bytes']
        #     # new_df['attainable_kernel_opsec'] = new_df[['peak_valu_flops','OI_HBM']].values.min(axis=1)
        #     # # new_df['attainable_kernel_opsec'] = min(new_df['curr_ai'] * new_df['HBM_bytes'], peak_valu_flops)
        #     # # new_df['HBM_BW'] = new_df['kernel_opsec'] / new_df['attainable_kernel_opsec']
        #     # self.calculate_bw(new_df, 'HBM_bytes', 'HBM_BW')

        #     # new_df = new_df.replace([np.inf,], np.nan, inplace=True)
        #     new_df.to_csv("./temp_2/" +  folder_name + '_1.csv')
        #     extended_individual_dfs[folder_name] = new_df      

        # individual_run_dfs.clear()


        # input_dir = "./temp_2/"
        # workloads = dict()

        
        # filelist = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if f.endswith('.csv')]
        # workloads[0] = filelist

        # output_prefix = []
        # # Add the desired path for the output, the system will automatically calculate the file name 
        # output_prefix.append(input_dir)

        # # print(workloads)

        # ## Reading and saving input
        # df_dict_input = {}
        # total_file_count = 0

        # for df_key in extended_individual_dfs.keys():
        #     df_original2 = extended_individual_dfs[df_key]
        #     # df_original2[df_original2.index.name] = df_original2.name
        #     df_original2['index'] = range(1, len(df_original2) + 1)
        #     df_original2 = df_original2.set_index('index')
        #     # print(df_original2.index.name)
        #     # print(df_original2['GRBM_GUI_ACTIVE'])

        #     ######### New Formulas here ######################
        #     df_original2['VALU_Util'] = 100 * (df_original2['SQ_ACTIVE_INST_VALU']/(df_original2['GRBM_GUI_ACTIVE'] * 104))
        #     df_original2['GPU_Activity'] = (df_original2['GRBM_GUI_ACTIVE'] / df_original2['GRBM_COUNT']) * 100
        #     df_original2['GPU_Occupancy'] = df_original2['SQ_ACCUM_PREV_HIRES'] / df_original2['GRBM_GUI_ACTIVE']
        #     df_original2['SALU_Util'] = 100 * (df_original2['SQ_ACTIVE_INST_SCA']/(df_original2['GRBM_GUI_ACTIVE'] * 104))
        #     df_original2['VALU_threads_per_wave_avg'] = df_original2['SQ_THREAD_CYCLES_VALU']/df_original2['SQ_ACTIVE_INST_VALU']
        #     df_original2['MFMA_Util'] = (100 * df_original2['SQ_VALU_MFMA_BUSY_CYCLES'])/(df_original2['GRBM_GUI_ACTIVE'] * 104)


        #     # print(df_original2['Kernel_Name'])
        #     df_original2['Kernel_Name']=[s.replace(',','_') for s in df_original2['Kernel_Name']]
        #     df_original2['Kernel_Name']=[s.replace(' ','_') for s in df_original2['Kernel_Name']]
        #     df_original2 = df_original2.groupby(['Kernel_Name']).agg(func='mean')
        #     df_dict_input[total_file_count] = df_original2.groupby(['Kernel_Name']).agg(func='mean')
        #     df_dict_input[total_file_count] = df_dict_input[total_file_count].reset_index()

        #     total_file_count += 1

        # # for ind in range(len(workloads)):
        # #     file_name_list = workloads[ind]
        # #     for fname in file_name_list:
        # #         if not os.path.exists(fname):
        # #             print(fname, ' does not exist....')
        # #             continue
        # #         else:
        # #             # print(total_file_count, fname)
        # #             df_original2 = pd.read_csv(fname)

        # #             ######### New Formulas here ######################
        # #             df_original2['VALU_Util'] = 100 * (df_original2['SQ_ACTIVE_INST_VALU']/(df_original2['GRBM_GUI_ACTIVE'] * 104))
        # #             df_original2['GPU_Activity'] = (df_original2['GRBM_GUI_ACTIVE'] / df_original2['GRBM_COUNT']) * 100
        # #             df_original2['GPU_Occupancy'] = df_original2['SQ_ACCUM_PREV_HIRES'] / df_original2['GRBM_GUI_ACTIVE']
        # #             df_original2['SALU_Util'] = 100 * (df_original2['SQ_ACTIVE_INST_SCA']/(df_original2['GRBM_GUI_ACTIVE'] * 104))
        # #             df_original2['VALU_threads_per_wave_avg'] = df_original2['SQ_THREAD_CYCLES_VALU']/df_original2['SQ_ACTIVE_INST_VALU']
        # #             df_original2['MFMA_Util'] = (100 * df_original2['SQ_VALU_MFMA_BUSY_CYCLES'])/(df_original2['GRBM_GUI_ACTIVE'] * 104)


        # #             print(df_original2['Kernel_Name'])
        # #             df_original2['Kernel_Name']=[s.replace(',','_') for s in df_original2['Kernel_Name']]
        # #             df_original2['Kernel_Name']=[s.replace(' ','_') for s in df_original2['Kernel_Name']]
        # #             df_original2 = df_original2.groupby(['Kernel_Name']).agg(func='mean')
        # #             df_dict_input[total_file_count] = df_original2.groupby(['Kernel_Name']).agg(func='mean')
        # #             df_dict_input[total_file_count] = df_dict_input[total_file_count].reset_index()
        # #             total_file_count += 1

        #             # print(total_file_count)

        # df_dict_copy = copy.deepcopy(df_dict_input)
        # ############## Calculating the standard deviation across indexed variables that are specific to channels.
        # for wl_id, wl in df_dict_copy.items():
        #     df2 = (pd.DataFrame.from_dict(wl)).filter(regex='\[*\]')#, orient='index', columns=['ColumnName']) 
        #     variables_with_indices = df2.columns
        #     count = {}
        #     base_varnames_list = []
        #     for each_var in variables_with_indices:
        #         base_varname = each_var.split('[')[0]
        #         if base_varname not in count:
        #             count[base_varname] = 1
        #             base_varnames_list.append(base_varname)
        #         else:
        #             count[base_varname] += 1

        #     new_wl = wl.drop(variables_with_indices, axis=1)

        #     for each_base_varname in base_varnames_list:
        #         df_each_base_varname = df2.filter(regex=each_base_varname)
        #         new_wl[each_base_varname+'_std'] = df_each_base_varname.std(axis=1)
        #     df_dict_copy[wl_id] = copy.deepcopy(new_wl)

        # # print([v for c, v in df_dict_copy.items()])
        # df_with_stdDev_vars = copy.deepcopy(df_dict_copy)

        # filter_list = ['BeginNs', 'EndNs', 'DispatchNs', 'BeginNs', 'EndNs', 'CompleteNs', 'Index','queue-id','queue-index', 'pid', 'tid', 'grd', 'wgr', 'lds', 'vgpr', 'sgpr', 'fbar', 'sig', 'obj']
        # target_list = ['runtime']

        # selected_df_dict = copy.deepcopy(df_with_stdDev_vars)

        # common_top_region_list = []
        # df_dict_updated = {}
        # cumulative_workload_id = 0
        # feature_list = {} 
        # region_list = {} 
        # shorten_region_list = {}
        # model_index = 0 

        # for model_index in range(len(workloads)):
            
        #     df_subdir = {}
        #     for wl in range(len(workloads[model_index])): #For each input parameter in a model
        #         df_subdir[wl] = selected_df_dict[cumulative_workload_id].copy()
        #         cumulative_workload_id += 1

        #     #Find the common regions and features. Should be consistent within a model.
        #     ## TODO: Make a dictionary of feature list per model
        #     feature_list[model_index], region_list[model_index] = self.find_common_features_and_regions(df_subdir, filter_list, "Kernel_Name")
        #     shorten_region_list[model_index] = region_list[model_index]
            
        #     cumulative_df = pd.DataFrame(columns=feature_list[model_index])
            
        #     for wl in range(len(workloads[model_index])): #For each input parameter in a model
        #         #read in the file
        #         df_original2 = df_subdir[wl].copy() #The dataset is already there. 
        #         ## REMOVE all columns that contain only 0s
        #         df_original2 = df_original2.loc[:, (df_original2 != 0).any(axis=0)]

        #         common_top_region_list.append([reg for reg in region_list[model_index]])

        
        #         # Now, appending all rows from different workloads to a stack workloads. Concat seems to be the way.
        #         # cumulative_df = cumulative_df.append(df_original2, ignore_index = True)
        #         cumulative_df = pd.concat([cumulative_df,df_original2], ignore_index=True, sort=False)
        #         # print(wl, df_original2['MemUnitBusy'])        

        #     df_dict_updated[model_index] = cumulative_df
        
        
        # # for numx in df_dict_updated[0]['MemUnitBusy']: 
        # #     print(len(df_dict_updated[0]['MemUnitBusy']), numx)

        # # print(common_top_region_list[0])
        # common_region_list = list(set(common_top_region_list[0]))
        # # print(common_region_list)
        # pathlib.Path("./temp/output").mkdir(parents=True, exist_ok=True)
        # for model_index in range(len(df_dict_updated)):#range(len(workloads)): #For each model
        #     df_original2 = df_dict_updated[model_index]
        #     for target in target_list:
        #         df_original3 = df_original2[feature_list[model_index]]
        #         ##Make sure to fill up all Nan values with 0
        #         df_original3 = df_original3.fillna(0)

        #         #Add a target at a time to original3
        #         df_original3[target] = df_original2[target]
        #         outfilename = './temp/output/final_' + target+'.csv'
        #         print('Generating ' + outfilename + '...')
        #         #generate_dash_data_format(df_original3, feature_list, target, region_list, outfilename, "KernelName")
        #         ## TODO: PASS the following function feature_list[ind] and region_list[ind]
        #         self.generate_dash_data_format(df_original3, feature_list[model_index], target, region_list[model_index], outfilename, "Kernel_Name")
                
        #         #generate_config_block_for_this_input(config_list[model_index] + '_'+target, outfilename, target)
        # print("Done generating dash format")

        # drvr = driver()
        # # print(os.getcwd())

        # print(user_target)
        # # file_name = '/home/cup7/omni_inte/omniperf/src/omniperf_visualize/dashing/configs/omni_inte.yml'
        # file_name = '/home/mohammad/omni_inte/src/omniperf_visualize/dashing/configs/omni_inte.yml'
        # with open(file_name, 'w') as txtfile:
        #     for target in user_target:
        #         s = str(target) + ':'
        #         txtfile.write(s + '\n')
        #         # s = '  data: /home/cup7/omni_inte/omniperf/build/temp/output/final_runtime.csv'
        #         s = '  data: /home/mohammad/omni_inte/build/temp/output/final_runtime.csv'
        #         txtfile.write(s + '\n')
        #         s = '  tasks:'
        #         txtfile.write(s + '\n')
        #         s = '    - modules.resource_score.compute_rsm_task_all_regions'
        #         txtfile.write(s + '\n')
        #         s = '    - viz.sunburst3.sunburst'
        #         txtfile.write(s + '\n')
        #         s = '  name:  \'' + target +'\''
        #         txtfile.write(s + '\n')
        #         s = '  target:  \'' + target +'\''
        #         txtfile.write(s + '\n')
        #         if inverse=='false':
        #             s = '  compute_target: modules.compute_target.compute_runtime'
        #             txtfile.write(s + '\n')
        #         else:
        #             s = '  compute_target: modules.compute_target.compute_inverse_target'
        #             txtfile.write(s + '\n')                
        #         s = '##############################'
        #         txtfile.write(s + '\n')

        #     txtfile.write('\n')

        #     s = 'main:'
        #     txtfile.write(s + '\n')  
        #     s = '  tasks:'
        #     txtfile.write(s + '\n')
        #     for target in user_target:
        #         s = '    - ' + str(target)
        #         txtfile.write(s + '\n')
        #     s = '    - viz.dashboard.dashboard_init'
        #     txtfile.write(s + '\n')
            
        #     s = '  arch: amd-mi200' + '\n'
        #     s += '  data_rescale: true\n'
        #     s += '  rsm_iters: 500\n'
        #     s += '  rsm_print: false\n'
        #     s += '  rsm_use_nn_solver: true\n'
        #     # s += '  save_compat: true\n'
        #     s += '  use_belief: true\n'
        #     s += '  compat_labels: true\n'
        #     s += '  shorten_event_name: false\n'
        #     s += '  port: ' + port + '\n'
        #     txtfile.write(s)

        # # if user_target=='SALU Util':
        # #     print('SALU UTIL')
        # #     omniperf = Omniperf()
        # #     omniperf.run_analysis()
        # drvr.main(file_name, True)

