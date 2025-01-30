"""
WPuQ PV Generation at Inverter (multiple households)
"""
import os
from functools import partial
import multiprocessing as mp
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import torch
from einops import rearrange

from .utils import TimeSeriesDataset, DatasetWithMetadata, NAME_SEASONS, PIT, months_of_season
from .heat_pump import shuffle_array, WPuQ
from energydiff.utils.io import WPuQPVReader

PREPROCESS_RES = '1min'
DIRECTION_CODE = {
    'EAST': 0,
    'SOUTH': 1,
    'WEST': 2,
}
RES_SECOND = {
    '1min': 60,
    '15min': 15*60,
    '30min': 30*60,
    '1h': 60*60,
    '60min': 60*60,
}

def get_first_last_moment_of_month(year: int, month: int, res_second: int = 10, offset: timedelta = timedelta(seconds=0)):
    first_day = datetime(year, month, 1, 0, 0, 0) + offset
    last_day = datetime(year, month, 1, 0, 0, 0) + relativedelta(months=1) - timedelta(seconds=res_second) + offset
    
    return first_day, last_day

def process_pv_dataset(list_dataset, year: int) -> dict[str, list[np.array]]:
    out = {str(month): [] for month in range(1, 13)}
    for full_dataset in list_dataset: # full == all three directions
        # dataset: custom-dtype array (p * 365,) with many fields
        all_dirs = np.unique(full_dataset['DIRECTION'])
        for dir_u10 in all_dirs:
            dataset = full_dataset[full_dataset['DIRECTION'] == dir_u10]
            dir_int = DIRECTION_CODE[dir_u10]
            x = np.sort(dataset['index']) # timestamps
            offset = datetime.fromtimestamp(x[0]) - datetime(year, 1, 1, 0, 0, 0)
            # res_second = 24*60*60 // (dataset.shape[0] // 365)
            res_second = RES_SECOND[PREPROCESS_RES]
            y = dataset['P_TOT'] # power consumption
            # interpolate
            print(f'***{year} {dir_u10}***')
            print(f'start: {datetime.fromtimestamp(x[0])}')
            print(f'end: {datetime.fromtimestamp(x[-1])}')
            # xp = np.linspace(
            #     datetime.fromtimestamp(x[0]).timestamp(),
            #     (datetime.fromtimestamp(x[0])+relativedelta(years=1)-timedelta(seconds=res_second)).timestamp(),
            #     num=365*24*60*60//res_second,
            # )
            # yp = np.interp(xp, x, y)
            
            for month in range(1, 13):
                first, last = get_first_last_moment_of_month(year, month, res_second, offset)
                xp = np.linspace(
                    int(first.timestamp()),
                    int(last.timestamp()),
                    int((last - first).total_seconds()) // res_second + 1
                )
                ds = np.interp(xp, x, y)
                ds = rearrange(ds, '(days per_day) -> days per_day', per_day=24*60*60//res_second)
                _dir = np.full((ds.shape[0], 1), dir_int).astype(np.float64)
                _dtype = np.dtype([
                    ('P_TOT', 'float64', (ds.shape[1],)),
                    ('DIRECTION', 'float64', (1,))
                ])
                structured_array = np.empty(ds.shape[0], dtype=_dtype)
                structured_array['P_TOT'] = ds
                structured_array['DIRECTION'] = _dir
                
                out[str(month)].append(structured_array)
            
    return out

class PreWPuQPV():
    """
    __all_fields__ = [('index', '<i8'), ('S_1', '<f8'), ('S_2', '<f8'), 
        ('S_3', '<f8'), ('S_TOT', '<f8'), ('I_1', '<f8'), 
        ('I_2', '<f8'), ('I_3', '<f8'), ('PF_1', '<f8'), 
        ('PF_2', '<f8'), ('PF_3', '<f8'), ('PF_TOT', '<f8'), 
        ('P_1', '<f8'), ('P_2', '<f8'), ('P_3', '<f8'), 
        ('P_TOT', '<f8'), ('Q_1', '<f8'), ('Q_2', '<f8'), 
        ('Q_3', '<f8'), ('Q_TOT', '<f8'), ('U_1', '<f8'), 
        ('U_2', '<f8'), ('U_3', '<f8')]
        
    _data_{res}.hdf5:
        - MISC
            - ES1
                - TRANSFORMER
                    - index
                    - P_TOT
                    
    _data_spatial.hdf5:
        - SUBSTATION
            - {res}
                - -
                    - index
                    - P_TOT
    
    """
    hdf5_suffix = {
        '10s': '_data_10s.hdf5',
        '1min': '_data_1min.hdf5',
        '15min': '_data_15min.hdf5',
        '60min': '_data_60min.hdf5',
        '1h': '_data_60min.hdf5',
    }[PREPROCESS_RES]
    col_names = [
        'index',
        'PF_TOT',
        'P_TOT',
        'DIRECTION',
    ]
    
    def __init__(
        self,
        root: str = 'data/wpuq/raw',
        year: int = 2018,
    ):
        self.root = root
        self.year = year
        self.reader = WPuQPVReader(os.path.join(root, f'{year}{self.hdf5_suffix}'),
                                      column_names=self.col_names)
        
    def load_process_save(self, num_process = 1):
        final_dataset_by_month = {
            str(month): None for month in range(1, 13)
        }
        train_dataset_by_month, val_dataset_by_month, test_dataset_by_month = {}, {}, {}
        train_num_sample_per_month, val_num_sample_per_month, test_num_sample_per_month = {}, {}, {}
                    
        with self.reader as reader:
            all_dataset = [dataset for dataset in reader]
            num_dataset_per_process = len(all_dataset) // num_process 
            list_dataset_per_process = [all_dataset[i*num_dataset_per_process:(i+1)*num_dataset_per_process] for i in range(num_process)] 
            list_dataset_per_process[-1] = list_dataset_per_process[-1] + all_dataset[num_dataset_per_process*num_process:]
            _proc_dataset = partial(process_pv_dataset, year=self.year)
            # pool = mp.Pool(num_process)
            list_dataset_per_process = list(map(_proc_dataset, list_dataset_per_process))
            # list_dataset_per_process = [process_dataset(list_dataset_per_process[i], self.year) for i in range(num_process)]
            # pool.close()
            # pool.join()
            for month in range(1, 13):
                # TODO: check if the process results are non empty in each month. also afterwards only save those months that are nonempty
                collected = []
                for idx_process in range(num_process):
                    item = list_dataset_per_process[idx_process][str(month)]
                    if len(item) > 0:
                        collected.append(np.concatenate(item, axis=0)) # collected: list[np.ndarray[day, seq_length]]
                if len(collected) == 0:
                    print(f'{self.year}-{month} is empty.')
                    continue
                final_dataset_by_month[str(month)] = shuffle_array(
                    np.concatenate(
                        collected,
                        axis=0
                    )
                )
                train_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][:int(len(final_dataset_by_month[str(month)])*0.5)]
                val_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][int(len(final_dataset_by_month[str(month)])*0.5):int(len(final_dataset_by_month[str(month)])*0.75)]
                test_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][int(len(final_dataset_by_month[str(month)])*0.75):]
                train_num_sample_per_month[str(month)] = len(train_dataset_by_month[str(month)])
                val_num_sample_per_month[str(month)] = len(val_dataset_by_month[str(month)])
                test_num_sample_per_month[str(month)] = len(test_dataset_by_month[str(month)])
        
        np.savez_compressed(os.path.join(self.root, 'wpuq_pv_'+str(self.year)+'_train.npz'), **train_dataset_by_month)
        np.savez_compressed(os.path.join(self.root, 'wpuq_pv_'+str(self.year)+'_val.npz'), **val_dataset_by_month)
        np.savez_compressed(os.path.join(self.root, 'wpuq_pv_'+str(self.year)+'_test.npz'), **test_dataset_by_month)
        
        print('complete.')
        return train_num_sample_per_month, val_num_sample_per_month, test_num_sample_per_month
    
class WPuQPV(WPuQ):
    common_prefix = 'wpuq_pv'
    base_res_second = 60 # base resolution = 60s
    
    def create_dataset(self) -> DatasetWithMetadata:
            # B: if processed not exists or load is False: load, clean, shuffle, vectorize, save
            raw_array = {}
            for task in self.record_tasks:
                raw_array[task] = []
                for year in self.record_years:
                    raw_array[task].append(np.load(os.path.join(self.raw_dir, self.common_prefix+'_'+str(year)+'_'+task+'.npz')))
            
            raw_array_collected = {}
            for task in self.record_tasks:
                raw_array_collected[task] = {}
                for month in range(1, 13):
                    raw_array_collected[task][str(month)] = []
                    for npz in raw_array[task]:
                        if str(month) in npz:
                            raw_array_collected[task][str(month)].append(npz[str(month)])
                    if len(raw_array_collected[task][str(month)]) > 0:
                        raw_array_collected[task][str(month)] = np.concatenate(raw_array_collected[task][str(month)], axis=0)
            
            # before processing, put all tensors together
            num_sample_task_season = []
            all_tensor = []
            all_dir = []
            for task in self.record_tasks:
                for season in NAME_SEASONS:
                    _profile_to_append = torch.from_numpy(
                        np.concatenate(
                            [raw_array_collected[task][str(month)]['P_TOT'] for month in months_of_season(season)],
                            axis=0
                        ).astype(np.float32)
                    )
                    _dir_to_append = torch.from_numpy(
                        np.concatenate(
                            [raw_array_collected[task][str(month)]['DIRECTION'] for month in months_of_season(season)],
                            axis=0
                        ).astype(np.float32)
                    )
                    # !! remove inf and nan
                    _profile_to_append, indices = self.clean_dataset(_profile_to_append)
                    _dir_to_append = _dir_to_append[indices]
    
                    all_tensor.append(_profile_to_append) # shape: [num_sample, seq_length]
                    all_dir.append(_dir_to_append) # shape: [num_sample, 1]
                    num_sample_task_season.append(len(_profile_to_append))
            all_tensor = torch.cat(all_tensor, dim=0) # shape: [num_sample, seq_length]
            all_tensor = rearrange(all_tensor, 'n l -> n () l') # shape: [num_sample, 1, seq_length]
            all_dir = torch.cat(all_dir, dim=0) # shape: [num_sample, 1]
            all_dir = rearrange(all_dir, 'n c -> n c 1') # shape: [num_sample, 1, 1]
            
            # resolution adjustment
            resolution = self.process_option['resolution']
            if resolution == '10s':
                pass
            else:
                if resolution == '1min':
                    _pool_kernel_size = 60 // self.base_res_second
                elif resolution == '15min':
                    _pool_kernel_size = 15*60 // self.base_res_second
                elif resolution == '30min':
                    _pool_kernel_size = 30*60 // self.base_res_second
                elif resolution == '1h':
                    _pool_kernel_size = 60*60 // self.base_res_second
                else:
                    raise NotImplementedError
                all_tensor = torch.nn.functional.avg_pool1d(all_tensor, kernel_size=_pool_kernel_size, stride=_pool_kernel_size)
            
            # normalize
            scaling_factor = None
            if self.process_option['normalize']:
                all_tensor, scaling_factor = self.normalize_fn(all_tensor)
                
            # pit
            pit = None
            if self.process_option['pit_transform']:
                pit = PIT(all_tensor[:sum(num_sample_task_season[:8])]) # only train and val
                all_tensor = self.pit.transform(all_tensor)
                
            # shuffle
            pass # already shuffled in pre-processing
        
            # vectorize
            if self.process_option['vectorize']:
                all_tensor = self.vectorize_fn(
                    all_tensor,
                    style=self.process_option['style_vectorize'],
                    window_size=self.process_option['vectorize_window_size']
                )
                
            # split
            task_season_chunk = list(all_tensor.split(num_sample_task_season, dim=0)) # shape: [num_sample, 1, seq_length]
            dir_task_season_chunk = list(all_dir.split(num_sample_task_season, dim=0)) # shape: [num_sample, 1, 1]
            profile_task_season = {}
            condition_task_season = {}
            for task in self.record_tasks:
                profile_task_season[task] = {}
                condition_task_season[task] = {}
                for idx, season in enumerate(NAME_SEASONS):
                    profile_task_season[task][season] = task_season_chunk.pop(0)
                    _season_label = torch.ones(profile_task_season[task][season].shape[0], dtype=torch.long) * idx # shape: [num_sample,]
                        # data type will be processed later when used. 
                    _season_label = rearrange(_season_label, 'n -> n 1 1')
                    _dir_label = dir_task_season_chunk.pop(0)
                    condition_task_season[task][season] = torch.cat([_season_label, _dir_label], dim=1) # shape: [num_sample, 2, 1]

            dataset = DatasetWithMetadata(
                profile = profile_task_season, # (b, r, T/r) profile
                label = condition_task_season, # (b, 2, 1) season and direction
                pit = pit,
                scaling_factor = scaling_factor
            )
            return dataset