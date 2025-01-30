import numpy as np

class HDF5Reader:
    """Read specific columns from a HDF5 file. (only NO_PV)
    
    Iterator: each time returns a structured array with the specified columns.
    
    Notes:
        - `SFH` is the case numer (e.g. SFH10)
    """
    def __init__(self, filename, column_names):
        self.filename = filename
        self.column_names = column_names
    
    def __enter__(self):
        import h5py
        self.f = h5py.File(self.filename, 'r')
        self.SFHs = list(self.f['NO_PV'].keys())
        self.SFHs = [SFH for SFH in self.SFHs if 'HEATPUMP' in self.f['NO_PV'][SFH].keys()]
        return self
    
    def __exit__(self, *args):
        self.f.close()
        
    def __iter__(self):
        self.SFH_index = 0
        return self
        
    def __next__(self):
        if self.SFH_index >= len(self.SFHs):
            raise StopIteration
        SFH = self.SFHs[self.SFH_index]
        table = self.f['NO_PV'][SFH]['HEATPUMP']['table']
        table = np.array(table)
        out = table[self.column_names]
        self.SFH_index += 1
        return out
    
    def __len__(self):
        return len(self.SFHs)
    
class WPuQTrafoReader:
    """Read specific columns from a HDF5 file. (only MISC/Transformer)
    
    Iterator: each time returns a structured array with the specified columns.
    
    f = h5py.File(...)
    f['MISC']['ES1']['TRANSFORMER']['table']
    
    """
    def __init__(self, filename, column_names):
        self.filename = filename
        self.column_names = column_names
    
    def __enter__(self):
        import h5py
        self.f = h5py.File(self.filename, 'r')
        return self
    
    def __exit__(self, *args):
        self.f.close()
        
    def __iter__(self):
        self.iter_idx = 0
        return self
        
    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration
        
        table = self.f['MISC']['ES1']['TRANSFORMER']['table']
        table = np.array(table)
        out = table[self.column_names] # shape (T,)
        self.iter_idx += 1
        return out
    
    def __len__(self):
        return 1 # only one transformer
    
class WPuQPVReader:
    """Read specific columns from a HDF5 file. (only MISC/Transformer)
    
    Iterator: each time returns a structured array with the specified columns.
    
    f = h5py.File(...)
    f['MISC']['ES1']['TRANSFORMER']['table']
    
    """
    directions = ['EAST', 'WEST', 'SOUTH'] # no north data. 
    def __init__(self, filename, column_names):
        self.filename = filename
        self.column_names = column_names
    
    def __enter__(self):
        import h5py
        self.f = h5py.File(self.filename, 'r')
        return self
    
    def __exit__(self, *args):
        self.f.close()
        
    def __iter__(self):
        self.iter_idx = 0
        return self
        
    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration
        
        tables = []
        for direction in self.directions:
            table = self.f['MISC']['PV1']['PV']['INVERTER'][direction]['table']
            table = np.array(table)
            # add extra column indicating the direction
            direction_col = np.full((table.shape[0],), direction, dtype='U10') # U10: unicaode string length 10
            new_dtype = np.dtype(table.dtype.descr + [('DIRECTION', 'U10')])
            # create a new array
            new_table = np.empty(table.shape, dtype=new_dtype)
            # copy data from the old table
            for field in table.dtype.names:
                new_table[field] = table[field]
            new_table['DIRECTION'] = direction_col
            tables.append(new_table[self.column_names])
        
        tables = np.concatenate(tables, axis=0)
        self.iter_idx += 1
        return tables
    
    def __len__(self):
        return len(self.directions) # three directions
    
def main():
    HDF5_PREFIX = '2019_data_15min'
    COL_NAMES = [
        'index', # timestamp
        'PF_TOT', # power factor total
        'P_TOT', # active power total
    ]
    reader = HDF5Reader('data/'+HDF5_PREFIX+'.hdf5', COL_NAMES)
    with reader as f:
        for index, table in enumerate(f):
            print(f"SFH count: {index+1},\t Shape: {table.shape}.")
            print(f"Column Names: {table.dtype.names}")
            print(f"First Row: {table[0]}")
    
def main():
    HDF5_PREFIX = '2019_data_15min'
    COL_NAMES = [
        'index', # timestamp
        'PF_TOT', # power factor total
        'P_TOT', # active power total
    ]
    reader = HDF5Reader('data/'+HDF5_PREFIX+'.hdf5', COL_NAMES)
    with reader as f:
        for index, table in enumerate(f):
            print(f"SFH count: {index+1},\t Shape: {table.shape}.")
            print(f"Column Names: {table.dtype.names}")
            print(f"First Row: {table[0]}")
    
if __name__ == '__main__':
    main()