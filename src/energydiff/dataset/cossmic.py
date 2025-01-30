"""
Data pre-processing and data loadiing for CoSSMic dataset 
Author: Nan Lin
Date: 2024-01-11

dataset url: https://data.open-power-system-data.org/household_data/2020-04-15
"""
from typing import Annotated as Float, Callable, Sequence, Set
import os
from datetime import datetime, date

import torch
from torch import Tensor
import numpy as np
from einops import rearrange
import pandas as pd
from tqdm import tqdm

from energydiff.dataset.utils import TimeSeriesDataset, DatasetWithMetadata, \
    NAME_SEASONS, months_of_season, PIT, season_of_month

def shuffle(array: np.ndarray, random_state: int = 0) -> np.ndarray:
    "Shuffle the array along the first axis"
    rng = np.random.default_rng(random_state)
    return array[rng.permutation(array.shape[0])]

class PreCoSSMic():
    """
    Preprocessing CoSSMic dataest. load the .csv file (1min) and save the processed files as .npz to the same directory.
    The dataset fis must be stored in the 'raw' directory of the specified root folder.
    
    Pipeline:
    .csv -> .pkl -> .npz
    
    Output unit: Watt
    
    Output data types:
    - grid_import
        - industrial
            - month: [1, 2, ..., 12]
        - residential
            - month: [1, 2, ..., 12]
        - public (e.g. school) (not used)
    - pv
        - industrial
            - month: [1, 2, ..., 12]
        - residential
            - month: [1, 2, ..., 12]
        
    """
    original_interval = 1./60. # 1/60 hour
    raw_file_name = 'cossmic_household_data_1min_singleindex.csv'
    common_prefix = 'cossmic'
    def __init__(
        self,
        root: str = 'data/',
        load_pickle: bool = True, 
    ):
        self.root = root
        
        # Check if the preprocessed csv files exist
        if self.preprocessed_pickle_exists() and load_pickle:
            print('Preprocessed pickle files exist. Loading pre-processed pickle files.')
            dfs = self.load_pickle()
        else:
            print('Preprocessed pickle files do not exist or `load_pickle` is False. Loading original csv files and processing.')
            dfs = self.load_process()
            self.save_pickle(dfs)
        
        arrays = self.further_process(dfs)
        self.save_npz(arrays)
        print('Done')
        
    def load_process(self) -> dict:
        "load original csv and convert to dict of dataframes, values converted to kW. "
        df = pd.read_csv(os.path.join(self.root, 'raw', self.raw_file_name), low_memory=False)
        
        # Get grid_import and pv columns -> 2 x df
        grid_import_cols = [col for col in df.columns if 'grid_import' in col]
        pv_cols = [col for col in df.columns if 'pv' in col]
        
        # Process datetime, get year, month, day, minute fields
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], format='%Y-%m-%dT%H:%M:%SZ')
        df['year'] = df['utc_timestamp'].dt.year
        df['month'] = df['utc_timestamp'].dt.month
        
        df.sort_values(by='utc_timestamp', inplace=True)
        
        # Loop through each location and type 
        dfs = {'grid_import': {}, 'pv': {}}
        for data_type, cols in [('grid_import', grid_import_cols), ('pv', pv_cols)]:
            for col in cols: # iterate over different locations
                # Convert from cumulative kWh to interval's average Watt
                df[col] = df[col].diff() / self.original_interval * 1000. # convert to Watt
                    # first row would be zero but it will be removed later
                location = col.split('_')[2]
                if location not in dfs[data_type]:
                    dfs[data_type][location] = {}
                
                for year in df['year'].unique():
                    if year not in dfs[data_type][location]:
                        dfs[data_type][location][str(year)] = {}
                    
                    for month in df['month'].unique():
                        # Filter for specific year, month, and type
                        filtered_df = df[(df['year'] == year) & (df['month'] == month)][['utc_timestamp'] + cols]
                    
                        # Reshape data into the daily vectors
                        daily_vectors = filtered_df.groupby(filtered_df['utc_timestamp'].dt.date)[col].apply(lambda x: x.tolist())

                        daily_df = daily_vectors.reset_index().rename(columns={'utc_timestamp': 'date', col: 'values'})
                        daily_df = daily_df[~daily_df['values'].apply(lambda x: all(pd.isna(x)))] # remove all nan rows
                        daily_df['location'] = location
                        
                        # If not empty, add to dfs
                        if not daily_df.empty:
                            daily_df['values'] = daily_df['values'].apply(lambda x: np.array(x, dtype=np.float32)) 
                                # convert kWh (1/60 h) to kW
                            dfs[data_type][location][str(year)][str(month)] = daily_df 
        
        return dfs
    
    def save_pickle(self, dfs: dict) -> None:
        save_dir = os.path.join(self.root, 'raw')
        os.makedirs(save_dir, exist_ok=True)
        for data_type in dfs:
            for location in dfs[data_type]:
                for year in dfs[data_type][location]:
                    for month in dfs[data_type][location][year]:
                        # dataframe to save
                        df_to_save = dfs[data_type][location][year][month]
                        
                        # filename
                        filename = f'{self.common_prefix}_{data_type}_{location}_{year}_{month}.pkl'
                        save_path = os.path.join(save_dir, filename)
                        
                        # save
                        df_to_save.to_pickle(save_path)
                        
                        print(f'Saved {save_path}')
        
    @staticmethod
    def is_weekend(date: date) -> bool:
        "Check if the date is weekend"
        return datetime.weekday() >= 5
    
    def load_pickle(self) -> dict:
        "load pickle files and convert to dict of dataframes"
        loaded_dfs = {'grid_import': {}, 'pv': {}}
        source_dir = os.path.join(self.root, 'raw')
        for filename in os.listdir(source_dir):
            if filename.endswith(".pkl") and (filename.startswith('cossmic_grid_import') or filename.startswith('cossmic_pv')):
                # Extract fields
                *data_type, location, year, month = filename.replace('.pkl', '').split('_')[1:]
                if isinstance(data_type, list):
                    data_type = '_'.join(data_type)
                    
                # Load the dataframe
                df = pd.read_pickle(os.path.join(source_dir, filename))
                
                # Organize in nested dictionary
                if location not in loaded_dfs[data_type]:
                    loaded_dfs[data_type][location] = {}
                if year not in loaded_dfs[data_type][location]:
                    loaded_dfs[data_type][location][year] = {}

                # Add the dataframe to the dictionary
                loaded_dfs[data_type][location][year][month] = df
                print(f'Loaded {filename}')
        
        return loaded_dfs
    
    def further_process(self, dfs: dict) -> dict:
        """
        Further processing for the specific analysis. 
        - distinguish location type, do not distinguish the exact location.
        - distinguish year, month, do not distinguish the exact date.
        - split the data into train, validation, test. ratio see the readme. 
        
        Input: 
            - dict of dataframes
            
        Output:
            - dict of 2d numpy arrays
        """
        arrays = {'grid_import': {}, 'pv': {}}

        # Process each DataFrame in dfs
        for data_type in dfs:
            for location in dfs[data_type]:
                if 'industrial' in location:
                    location_type = 'industrial'
                elif 'residential' in location:
                    location_type = 'residential'
                elif 'public' in location:
                    location_type = 'public'
                else:
                    location_type = 'unknown'
                for year in dfs[data_type][location]:
                    for month in dfs[data_type][location][year]:
                        df = dfs[data_type][location][year][month]

                        # Filter out rows with less than 1440 dimensions or NaN values
                        df = df[df['values'].apply(lambda x: len(x) == 1440 and not any(pd.isna(x)))]
                        if df.empty:
                            continue
                        array = np.stack(df['values'].values)

                        # Merge arrays by location type
                        if location_type not in arrays[data_type]:
                            arrays[data_type][location_type] = {}
                        if year not in arrays[data_type][location_type]:
                            arrays[data_type][location_type][year] = {}
                        if month not in arrays[data_type][location_type][year]:
                            arrays[data_type][location_type][year][month] = array
                        else:
                            arrays[data_type][location_type][year][month] = np.concatenate((arrays[data_type][location_type][year][month], array), axis=0)

        # Shuffle and Split the arrays
        for data_type in arrays:
            for location_type in arrays[data_type]:
                for year in arrays[data_type][location_type]:
                    for month in arrays[data_type][location_type][year]:
                        array = arrays[data_type][location_type][year][month]

                        # Shuffle the array
                        array = shuffle(array, random_state=0)

                        # Split into train, val, test
                        train_size = int(len(array) * 0.5)
                        val_size = int(len(array) * 0.25)

                        arrays[data_type][location_type][year][month] = {
                            'train': array[:train_size],
                            'val': array[train_size:train_size+val_size],
                            'test': array[train_size+val_size:]
                        }

        return arrays
    
    @staticmethod
    def decorate_data_type(str):
        if str == 'grid_import':
            return 'grid-import'
        else:
            return str
    
    def save_npz(self, arrays: dict) -> None:
        "Save the arrays as npz files"
        save_dir = os.path.join(self.root, 'raw')
        os.makedirs(save_dir, exist_ok=True)
        for data_type in arrays:
            for location_type in arrays[data_type]:
                for year in arrays[data_type][location_type]:
                    for task in ['train', 'val', 'test']:
                        # Prepare data to be saved in .npz file
                        data_to_save = {}
                        for month in arrays[data_type][location_type][year]:
                            key = str(month)
                            data_to_save[key] = arrays[data_type][location_type][year][month][task]

                        # Check if there's any data to save
                        if data_to_save:
                            # Create filename
                            filename = f'{self.common_prefix}_{self.decorate_data_type(data_type)}_{location_type}_{year}_{task}.npz'
                            save_path = os.path.join(save_dir, filename)
                            # Save the data
                            np.savez_compressed(save_path, **data_to_save)
                            print(f'Saved {save_path}')
                            
    def preprocessed_pickle_exists(self):
        "Check if the preprocessed pkl files exist"
        source_dir = os.path.join(self.root, 'raw')
        for filename in os.listdir(source_dir):
            if filename.endswith(".pkl") and (filename.startswith('cossmic_grid_import') or filename.startswith('cossmic_pv')):
                return True
            
        return False
    
class CoSSMic(TimeSeriesDataset):
    """
    CoSSMic dataset.
    
    Version 1.0.0: unconditionial + season conditions
    contains two datasets: grid_import and pv 
    each contains two location types: industrial and residential
    """
    common_prefix = 'cossmic'
    recorded_datasts = {'grid-import_industrial', 'grid-import_residential', 'grid-import_public', 'pv_industrial', 'pv_residential'}
    recorded_years = {'2015', '2016', '2017', '2018', '2019'}
    recorded_tasks = {'train', 'val', 'test'}
    condition_mapping = {
        "year": {
            2015: 2015,
            2016: 2016,
            2017: 2017,
            2018: 2018,
            2019: 2019,
        },
        "month": {
            m: m for m in range(1, 13)
        },
        "area": {
            'industrial': 0,
            'residential': 1,
            'public': 2,
        },
        "season": {
            NAME_SEASONS[0]: 0,
            NAME_SEASONS[1]: 1,
            NAME_SEASONS[2]: 2,
            NAME_SEASONS[3]: 3,
        }
    }
    original_dict_cond_dim = {
        'year': 1,
        'month': 1,
        'season': 1,
        'area': 1,
    }
    def __init__(
        self,
        root: str = 'data/',
        # data_type: str | list[str] = 'grid_import',
        # area: str | list[str] = 'industrial',
        target_labels: None|Sequence[str] = None,
        load: bool = True,
        resolution: str = '1min',
        normalize: bool = True,
        pit_transform: bool = False, 
        shuffle: bool = True,
        vectorize: bool = False,
        style_vectorize: str = 'chronological',
        vectorize_window_size: int = 3,
        sub_dataset_names: None|Sequence[str] = None,
    ):
        super().__init__()
        self.root = root
        # self.selected_data_type = [data_type] if isinstance(data_type, str) else data_type
        # self.selected_area = [area] if isinstance(area, str) else area
        assert resolution in ['1min', '15min', '30min', '1h', '12h'], 'resolution must be one of [1min, 15min, 30min, 60min]'
        assert sub_dataset_names is not None
        if target_labels is None:
            target_labels = list(self.original_dict_cond_dim.keys())
        else:
            target_labels = list(target_labels)
            for target_label in target_labels:
                if target_label not in self.all_dict_cond_dim:
                    self.all_dict_cond_dim[target_label] = 1
        self.process_option = {
            'target_labels': target_labels,
            'resolution': resolution,
            'normalize': normalize,
            'pit_transform': pit_transform, 
            'shuffle': shuffle,
            'vectorize': vectorize,
            'style_vectorize': style_vectorize,
            'vectorize_window_size': vectorize_window_size
        }
        hashed_option = self.hash_option(self.process_option)

        self.sub_dataset_names = {
            dataset_name for dataset_name in self.recorded_datasts if dataset_name in sub_dataset_names
        } # dataset_name: dataset
        hashed_sub_dataset_names = self.hash_set_string(set(sub_dataset_names))
        
        self.dataset = None
        if load:
            self.dataset = self.load_dataset(hashed_sub_dataset_names, hashed_option)
            
        if self.dataset is not None:
            print('All processed data loaded.')
        else:          
            print('Process and save data.')
            self.dataset = self.create_dataset(self.sub_dataset_names)
            self.save_dataset(self.dataset, hashed_sub_dataset_names, hashed_option)
        
        print('Dataset ready.')
    
    def get_processed_filename(self, hashed_dataset_names: str, hashed_option: str) -> str:
        "Return the filename of the processed data"
        return f'{self.common_prefix}_{hashed_dataset_names}_{hashed_option}.pt'        
    
    def load_dataset(self, hashed_sub_dataset_names, hashed_option) -> DatasetWithMetadata:
        loaded = None
        processed_filename = self.get_processed_filename(hashed_sub_dataset_names, hashed_option)
        if os.path.exists(os.path.join(self.processed_dir, processed_filename)):
            try: 
                loaded = torch.load(os.path.join(self.processed_dir, processed_filename), map_location='cpu')
                print(f'Loaded processed data {processed_filename}. ')
            except:
                print(f'Error loading processed data {processed_filename}. ')
        return loaded
    
    def save_dataset(self, dataset: DatasetWithMetadata, hashed_sub_dataset_names, hashed_option) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        processed_filename = self.get_processed_filename(hashed_sub_dataset_names, hashed_option)
        torch.save(dataset, os.path.join(self.processed_dir, processed_filename))
        print(f'Saved {processed_filename}')
    
    def create_dataset(self, sub_dataset_names: Set[str]) -> DatasetWithMetadata:
        " note: earlier versions distinguish 'area', but now we fuse all areas together."
        raw_year_profile = {
            'train': [],
            'val': [],
            'test': [],
        }
        for sub_dataset_name in sub_dataset_names:
            filename_foo = f'{self.common_prefix}_{sub_dataset_name}'
            _dataset_type, _area = sub_dataset_name.split('_') # 'grid-import' _ 'industrial'
            # stack years
            for task in self.recorded_tasks:
                for year in self.recorded_years:
                    filename_bar = f'_{year}_{task}.npz'
                    load_npz_path = os.path.join(self.raw_dir, filename_foo + filename_bar)
                    if os.path.exists(load_npz_path):
                        raw_year_profile[task].append((
                            {
                                'year': int(year), 
                                'dataset_type': _dataset_type,
                                'area': _area,
                            }, 
                            np.load(load_npz_path)
                        )) # tuple of (labels, npz)

        # separate by month
        raw_profile_collected = {}
        raw_label_collected = {}
        for task in self.recorded_tasks:
            _list_raw_profile = []
            _list_raw_label = []
            for month in range(1, 13):
                for labels, npz in raw_year_profile[task]:
                    if str(month) in npz:
                        if len(npz[str(month)]) == 0:
                            continue
                        _list_raw_profile.append(npz[str(month)])
                        _list_label = {
                            'year': np.array(np.array([[labels['year'],]] * npz[str(month)].shape[0])), # (num_samples, 1)
                            'month': np.array(np.array([[month,]] * npz[str(month)].shape[0])), # (num_samples, 1)
                            'season': np.array(np.array([[NAME_SEASONS.index(season_of_month(month)),]] * npz[str(month)].shape[0])), # (num_samples, 1)
                            'area': np.array(np.array([[self.condition_mapping['area'][labels['area']],]] * npz[str(month)].shape[0])), # (num_samples, 1)
                        }
                        _list_raw_label.append(
                            np.concatenate(
                                [_list_label[label_name] for label_name in self.process_option['target_labels']], 
                                axis=1 # (num_samples, num_labels
                            )
                        )
            # for all months
            if len(_list_raw_profile) > 0:
                raw_profile_collected[task] = np.concatenate(_list_raw_profile, axis=0) # all years, all months
                raw_label_collected[task] = np.concatenate(_list_raw_label, axis=0)
                    
        assert 'train' in raw_profile_collected, 'No data for train.'
        assert 'val' in raw_profile_collected, 'No data for val.'
        assert 'test' in raw_profile_collected, 'No data for test.'
        # put all tasks together to process. 
        num_sample_task = [] # used later to split
        all_tensor = []
        all_label = []
        for task in self.recorded_tasks:
            _profile_to_append = torch.from_numpy(
                raw_profile_collected[task].astype(np.float32)
            )
            _label_to_append = torch.from_numpy(
                raw_label_collected[task].astype(np.float32)
            )
            all_tensor.append(_profile_to_append)
            all_label.append(_label_to_append)
            num_sample_task.append(_profile_to_append.shape[0])
        all_tensor = torch.cat(all_tensor, dim=0) # shape: (num_sample, num_features)
        all_tensor = rearrange(all_tensor, 'n l -> n () l') # shape: (num_sample, 1, seq_length)
        all_label = torch.cat(all_label, dim=0) # shape: (num_sample, 2)
        all_label = rearrange(all_label, 'n c -> n c ()') # shape: (num_sample, 2, 1)
        
        # resolution adjustment
        res = self.process_option['resolution']
        if res == '1min':
            pass
        else:
            if res == '15min':
                _pool_kernel_size = 15
            elif res == '30min':
                _pool_kernel_size = 30
            elif res == '1h':
                _pool_kernel_size = 60
            elif res == '12h':
                _pool_kernel_size = 720
            else:
                raise NotImplementedError
            all_tensor = torch.nn.functional.avg_pool1d(all_tensor, kernel_size=_pool_kernel_size, stride=_pool_kernel_size)
            
        # remove outliers
        _pit = PIT(all_tensor.reshape(-1,1), perturb=True)
        upper_bound = _pit.inverse_transform(torch.tensor([0.99999,])) # 1 - 1e-5
        del _pit
        # do the removal later. 
        
        # normalize
        scaling_factor = None
        if self.process_option['normalize']:
            all_tensor, scaling_factor = self.normalize_fn(all_tensor, max_value=upper_bound)
            
        # pit
        pit = None
        if self.process_option['pit_transform']:
            pit = PIT(all_tensor[:sum(num_sample_task[:2])], perturb=True) # only train and val
            all_tensor = pit.transform(all_tensor)
            
        # shuffle
        pass # already shuffled in the pre-processing step
    
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
            profile = profile_task_season, # dict[str(task), tensor]
            label = label_task_season,
            pit = pit,
            scaling_factor = scaling_factor,
        )
        
        # close npz files
        for task in self.recorded_tasks:
            for labels, npz in raw_year_profile[task]:
                npz.close()
        
        return dataset_with_metadata
    
    @property
    def prefix_all_dataset(self) -> dict:
        "Return the prefixes of all datasets. data_type + area = one dataset"
        prefix_datasets = {}
        for data_type in self.recorded_data_types:
            prefix_datasets[data_type] = {}
            for area in self.recorded_areas:
                prefix_datasets[data_type][area] = f'{self.common_prefix}_{data_type}_{area}'
                
        return prefix_datasets
                
    
    def load_npz(self) -> dict:
        num_samples = {}
        for data_type in self.recorded_data_types:
            num_samples[data_type] = {}
            for area in self.recorded_areas:
                num_samples[data_type][area] = {}
                for year in self.recorded_years:
                    num_samples[data_type][area][year] = {
                        str(month): {} for month in range(1, 13)
                    }
                    for task in self.recorded_tasks:
                        filename = f'{self.common_prefix}_{data_type}_{area}_{year}_{task}.npz'
                        load_path = os.path.join(self.root, 'raw', filename)
                        if os.path.exists(load_path):
                            with np.load(load_path) as data:
                                npzfile = dict(data)
                                for month in npzfile:
                                    num_samples[data_type][area][year][month][task] = npzfile[month].shape[0]
                        else:
                            # print(f'File does not exist: {load_path}')
                            pass
                            
        return num_samples
    
    def map_label(self, **kwargs) -> Tensor:
        labels = self.process_option['target_labels']
        label_tensor = torch.full((len(labels),), float('nan'))
        for idx, label in enumerate(labels):
            if label in kwargs:
                label_tensor[idx] = self.condition_mapping[label][kwargs[label]]
        
        return label_tensor
    
    def __repr__(self):
        info = f'CoSSMic dataset. Process option: {self.process_option}'
        for data_type in self.num_samples:
            data_type_total = 0
            for area in self.num_samples[data_type]:
                area_total = 0
                for year in self.num_samples[data_type][area]:
                    year_total = 0
                    for month in self.num_samples[data_type][area][year]:
                        month_total = sum(self.num_samples[data_type][area][year][month].values())
                        info += f'\n{data_type}_{area}_{year}_{month}: {month_total}'
                        year_total += month_total
                    info += f'\n{data_type}_{area}_{year}: {year_total}'
                    area_total += year_total
                info += f'\n{data_type}_{area}: {area_total}'
                data_type_total += area_total
            info += f'\n{data_type}: {data_type_total}'
        
        return info