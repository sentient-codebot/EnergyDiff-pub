import os
from functools import partial
import multiprocessing as mp
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
from einops import rearrange

from .utils import TimeSeriesDataset
from .heat_pump import shuffle_array, WPuQ
from energydiff.utils.io import WPuQTrafoReader

PREPROCESS_RES = '1min'
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

def process_trafo_dataset(list_dataset, year: int) -> dict[str, list[np.array]]:
    out = {str(month): [] for month in range(1, 13)}
    for dataset in list_dataset:
        # dataset: custom-dtype array (p * 365,) with many fields
        x = np.sort(dataset['index']) # timestamps
        offset = datetime.fromtimestamp(x[0]) - datetime(year, 1, 1, 0, 0, 0)
        # res_second = 24*60*60 // (dataset.shape[0] // 365)
        res_second = RES_SECOND[PREPROCESS_RES]
        y = dataset['P_TOT'] # power consumption
        # interpolate
        print(f'***{year}***')
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
            
            out[str(month)].append(ds)
            
    return out

class PreWPuQTrafo():
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
    ]
    
    def __init__(
        self,
        root: str = 'data/wpuq/raw',
        year: int = 2018,
    ):
        self.root = root
        self.year = year
        self.reader = WPuQTrafoReader(os.path.join(root, f'{year}{self.hdf5_suffix}'),
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
            _proc_dataset = partial(process_trafo_dataset, year=self.year)
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
                    ).astype(np.float32)
                )
                train_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][:int(len(final_dataset_by_month[str(month)])*0.5)]
                val_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][int(len(final_dataset_by_month[str(month)])*0.5):int(len(final_dataset_by_month[str(month)])*0.75)]
                test_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][int(len(final_dataset_by_month[str(month)])*0.75):]
                train_num_sample_per_month[str(month)] = len(train_dataset_by_month[str(month)])
                val_num_sample_per_month[str(month)] = len(val_dataset_by_month[str(month)])
                test_num_sample_per_month[str(month)] = len(test_dataset_by_month[str(month)])
        
        np.savez_compressed(os.path.join(self.root, 'wpuq_trafo_'+str(self.year)+'_train.npz'), **train_dataset_by_month)
        np.savez_compressed(os.path.join(self.root, 'wpuq_trafo_'+str(self.year)+'_val.npz'), **val_dataset_by_month)
        np.savez_compressed(os.path.join(self.root, 'wpuq_trafo_'+str(self.year)+'_test.npz'), **test_dataset_by_month)
        
        print('complete.')
        return train_num_sample_per_month, val_num_sample_per_month, test_num_sample_per_month
    
class WPuQTrafo(WPuQ):
    common_prefix = 'wpuq_trafo'
    base_res_second = 60 # base resolution = 60s
