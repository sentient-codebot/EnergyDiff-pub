"""
Version 0 Design: Divide 168 .csv files into [130, 19, 19] as train, val, test
"""

from typing import Annotated as Float
import os

import torch
from torch import Tensor
import numpy as np
from einops import rearrange
import pandas as pd
from tqdm import tqdm

from energydiff.dataset.utils import TimeSeriesDataset, PIT, DatasetWithMetadata

# helper functions
class PreLCL():
    """
    Preprocess LCL data. Load from 'raw' dir but also saves to 'raw' dir. Because this is just preprocessing. 
    The actuall processed folder is reserved for after normalization, vectorization, etc.
    """
    data_prefix = 'LCL-June2015v2_'
    data_suffix = '.csv'
    def __init__(
        self,
        root: str = 'data/',
        list_case: list[int] = [0],
    ):
        self.root = root
        self.list_case = list_case
        self.list_filename = [self.data_prefix + str(case) + self.data_suffix for case in self.list_case]
        
    def load_and_process_csv(self, file):
        df = pd.read_csv(file)

        # Filter rows where stdorToU is 'Std'
        df = df[df['stdorToU'] == 'Std']
        
        # Convert 'KWH/hh (per half hour) ' column to float
        df = df.fillna(np.nan)
        df['KWH/hh (per half hour) '] = pd.to_numeric(df['KWH/hh (per half hour) '], errors='coerce')

        # Convert DateTime column to datetime format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['Year-Month-Day'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['DateTime'].dt.day.astype(str)
        df['Week'] = df['DateTime'].dt.isocalendar().week
        df['Year-Week'] = df['Year'].astype(str) + '-' + df['Week'].astype(str)
        df['Day'] = df['DateTime'].dt.dayofweek
        df['Minute'] = df['DateTime'].dt.hour*60 + df['DateTime'].dt.minute # Minute of the week
        
        # Sort by LCLid, Year, Month, Day, Minute
        df = df.sort_values(by=['LCLid', 'Year', 'Month', 'Day', 'Minute'], ascending=[True, True, True, True, True])
        df_grouped = df.groupby(['LCLid', 'Year-Month-Day'])
        
        # Filter out incomplete weeks
        df_complete_days = df_grouped.filter(lambda x: x['Minute'].nunique() == 2*24)
        
        
        # Pivot the table to create a matrix with users on one axis and hours of the week on the other
        df_pivoted = df_complete_days.pivot_table(index=['LCLid', 'Year-Month-Day'], 
                                    columns='Minute', 
                                    values='KWH/hh (per half hour) ',
                                    ).reset_index() # fill_value not enabled. keep NaNs for now

        # Convert from kWh/hh to kW
        df_pivoted.iloc[:,2:] = df_pivoted.iloc[:,2:] / 0.5
        
        # Clean data
        df_pivoted.iloc[:,2:] = df_pivoted.iloc[:,2:].replace([np.inf, -np.inf], np.nan)
        df_pivoted = df_pivoted.dropna(subset=df_pivoted.columns[2:])

        return df_pivoted

    def load_process_save(self,):
        df_list = []
        for filename in tqdm(self.list_filename, total=len(self.list_filename)):
            df_list.append(self.load_and_process_csv(os.path.join(self.raw_dir, filename)))
            
        # Combine all dataframes
        str_case = '_'.join([str(self.list_case[0]), str(self.list_case[-1])])
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(os.path.join(self.preprocessed_dir, f'lcl_electricity_{str_case}.csv'), index=False)
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def preprocessed_dir(self):
        return os.path.join(self.root, 'raw')

# dataset class
class LCLElectricityProfile(TimeSeriesDataset):
    """
    Low Carbon London Electricity Dataset
    
    version 1.0.0: only unconditional + all year
    
    Data stored in attributes:
    self.dataset: DatasetWithMetadata
    
    """
    common_prefix = 'lcl_electricity'
    recorded_tasks = ['train', 'val', 'test']
    def __init__(
        self,
        root = 'data/',
        load = False,
        resolution: str = '30min',
        normalize = True,
        pit_transform = False,
        shuffle = True,
        vectorize = True,
        style_vectorize = 'chronological',
        vectorize_window_size = 3,
    ):
        self.root = root
        self.process_option = {
            'resolution': resolution,
            'normalize': normalize,
            'pit_transform': pit_transform,
            'shuffle': shuffle,
            'vectorize': vectorize,
            'style_vectorize': style_vectorize,
            'vectorize_window_size': vectorize_window_size,
        }
        
        self.hashed_option = self.hash_option(self.process_option)
        processed_filename = self.get_processed_name()
            
        # Branch A: if processed data exists, load it
        self.dataset = None
        if load:
            self.dataset = self.load_dataset(processed_filename)
        
        if self.dataset is None:
            print('Process and save data.')
            dataset = self.create_dataset()
            self.dataset = dataset
            self.save_dataset(dataset, processed_filename)
        else:
            print('All processed data loaded.')
            
        print('Dataset ready.')
        
    def get_processed_name(self):
        return f'{self.common_prefix}_processed_{self.hash_option(self.process_option)}.pt'
            
    def create_dataset(self):
        # Branch B: if processed data does not exist, load raw data and process it
        raw_path = {
            task: os.path.join(self.raw_dir, 'lcl_electricity_'+task+'.csv') for task \
                in self.recorded_tasks
        } # example: {'train':k '.../data/raw/lcl_electricity_train.pt'}

        # load raw dataframes
        dfs = {}
        for task, path in raw_path.items():
            try:
                dfs[task] = pd.read_csv(path)
            except FileNotFoundError:
                raise FileNotFoundError(f'File {path} not found. Please run PreLCL().load_process_save() first.')
        
        # extract data from dataframe
        all_labels = {} # train/val/test
        all_profiles = {} # train/val/test
        for task in dfs.keys():
            # label
            year_month_day = dfs[task]['Year-Month-Day']
            year = year_month_day.str.split('-', expand=True)[0].astype(int)
            month = year_month_day.str.split('-', expand=True)[1].astype(int)
            day = year_month_day.str.split('-', expand=True)[2].astype(int)
            year = torch.tensor(year.values, dtype=torch.float32).unsqueeze(1) # shape [b, 1]
            month = torch.tensor(month.values, dtype=torch.float32).unsqueeze(1) # shape [b, 1]
            day = torch.tensor(day.values, dtype=torch.float32).unsqueeze(1) # shape [b, 1]

            all_labels[task] = torch.cat([year, month, day], dim=1) # shape [b, 3]
            # profile
            profile = dfs[task].iloc[:, 2:].values # shape [b, l]
            profile = torch.tensor(profile, dtype=torch.float32) # shape [b, l]
            assert profile.shape[1] == 48
            all_profiles[task] = profile
            
        # prep for processing
        all_tensor = torch.cat([
                all_profiles['train'],
                all_profiles['val'],
                all_profiles['test'],
        ], dim=0) # shape [b, c]
        all_tensor = rearrange(all_tensor, 'b l -> b () l') # shape [b, 1, l]
        all_label = torch.cat([
            all_labels['train'],
            all_labels['val'],
            all_labels['test'],
        ], dim=0) # shape [b, 3]
        all_label = rearrange(all_label, 'b c -> b c ()') # shape [b, 3, 1]
        num_sample_task = [
            all_profiles['train'].shape[0],
            all_profiles['val'].shape[0],
            all_profiles['test'].shape[0],
        ]
        
        # resolution adjustment
        res = self.process_option['resolution']
        if res == '30min':
            pass
        else:
            if res == '1h':
                _pool_kernel_size = 2
            elif res == '6h':
                _pool_kernel_size = 12
            elif res == '12h':
                _pool_kernel_size = 24
            elif res == '1d' or res == '24h':
                _pool_kernel_size = 48
            else:
                raise NotImplementedError
            all_tensor = torch.nn.functional.avg_pool1d(all_tensor, kernel_size=_pool_kernel_size, stride=_pool_kernel_size)
        
        # remove outliers
        _pit = PIT(all_tensor.reshape(-1,1), perturb=True)
        upper_bound = _pit.inverse_transform(torch.tensor([0.99999,])) # 1 - 1e-5
        del _pit
        # do the removal later
        
        # normalize
        scaling_factor = None
        if self.process_option['normalize']:
            all_tensor, scaling_factor = self.normalize_fn(all_tensor)
            
        # pit
        pit = None
        if self.process_option['pit_transform']:
            pit = PIT(all_tensor[:sum(num_sample_task[:2])], perturb=True)
            all_tensor = pit.transform(all_tensor)
        
        # shuffle
        pass # let dataloader do the shuffling
        
        # vectorize
        if self.process_option['vectorize']:
            all_tensor = self.vectorize_fn(
                all_tensor, 
                style=self.process_option['style_vectorize'],
                window_size=self.process_option['vectorize_window_size']
            )
        
        # split
        profile_task_chunk = list(all_tensor.split(num_sample_task, dim=0)) # len = 3, train, val, test
        label_task_chunk = list(all_label.split(num_sample_task, dim=0))
        profile_task_season = {}
        label_task_season = {}
        for task in self.recorded_tasks:
            profile_task_season[task] = profile_task_chunk.pop(0)
            label_task_season[task] = label_task_chunk.pop(0)
            
        # post-outlier removal
        for task in self.recorded_tasks:
            indices = torch.all(profile_task_season[task].flatten(start_dim=1) <= 1., dim=-1)
            profile_task_season[task] = profile_task_season[task][indices]
            label_task_season[task] = label_task_season[task][indices]
            
        # summarize
        dataset_with_metadata = DatasetWithMetadata(
            profile=profile_task_season,
            label=label_task_season,
            pit=pit,
            scaling_factor=scaling_factor,
        )
        return dataset_with_metadata
        
    def save_dataset(self, dataset, processed_filename):
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(dataset, os.path.join(self.processed_dir, processed_filename))
        print(f'Saved {processed_filename}')
        
        
    def load_dataset(self, processed_filename):
        if os.path.exists(os.path.join(self.processed_dir, processed_filename)):
            try:
                loaded: DatasetWithMetadata = torch.load(os.path.join(self.processed_dir, processed_filename))
                return loaded
            except:
                print('Error loading processed data. Recreating...')
                return None