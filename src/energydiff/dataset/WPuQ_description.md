# WPuQ Heat Pump Dataset

website: https://zenodo.org/records/5642902

paper: https://www.nature.com/articles/s41597-022-01156-1

format: hdf5

unit: 
- "S": "VA"
- "P": "W"
- "Q": "VAR"
- "PF": "no unit"
- "U": "V"
- "I": "A"

measurements:
- heat pump power consumption
- household power consumption
- (optional) PV power generation

resolution: 10s (original), 15min and 30min (aggregated)

scope: spanning three years from 2018 to 2020. some households have missing data. 

spatial: ~40 households. 

subjects:
- households. marked as "SFHXX" in the dataset, SFH stands for single family households. XX is an index for different households. 
- with or without PV panels. With-PV cases can have negative active power consumption. 

## Preprocessed Files

Contains only the 'P_TOT' key, which is the active power consumption of the heat pump. 

- 'wpuq_{YEAR}_{TASK, train/val/test}.npz'
  - In each npz file, there are up to 12 keys (and their according arrays), which is the month index, as strings. 
  - 2018: includes **only** months from 5 to 12
  - 2019: includes all months from 1 to 12
  - 2020: includes all months from 1 to 12

- shape of each array: [N, 8640] (8640 = 24\*60\*60/10), i.e. 10s resolution. N is the number of samples. 

If coarser resolution is needed:
- 1 min, 1440-dim: take mean for every 6 elements
- 15 min, 96-dim: take mean for every 90 elements (6\*15)
- 30 min, 48-dim: take mean for every 180 elements (6\*30)
- 1 hour, 24-dim: take mean for every 360 elements (6\*60)

## Preprocessing

We keep the year and month information, and divide the data into days and see them as vectors (i.e. daily consumption profiles). We do not distinguish different user anymore. **Therefore SFHXX information is lost in this preprocessing. The exact day of the month is also lost.** Technically, we
- removed missing values
- reshaped to daily data, i.e. [samples, $24*60*6$]. 
- shuffled over different days and households within the same year and month. 
- partitioned into train/val/test sets.

### Train/val/test Partitioning

Train/val/test partitioning ratio: 0.5/0.25/0.25

Number of samples of of each task and year:

- Month-agnostic
  - train: 3393 + 5556 + 5125 = 14074
  - val: 1697 + 2777 + 2563 = 7037
  - test: 1700 + 2783 + 2567 = 7050

- All three years divided by month (training)
  - 0 + 495 + 450 = 945
  - 0 + 462 + 435 = 897
  - 0 + 491 + 450 = 941
  - 0 + 480 + 442 = 922
  - 246 + 484 + 435 = 1165
  - 428 + 465 + 405 = 1298
  - 435 + 475 + 418 = 1328
  - 448 + 464 + 408 = 1320
  - 450 + 435 + 395 = 1280
  - 436 + 427 + 421 = 1284
  - 449 + 420 + 432 = 1301
  - 501 + 458 + 434 = 1393
  - total = 14074

- 2018:
  - train: total = 3393
    - {'5': 246, '6': 428, '7': 435, '8': 448, '9': 450, '10': 436, '11': 449, '12': 501}
  - val: total = 1697
    - {'5': 123, '6': 214, '7': 218, '8': 224, '9': 225, '10': 218, '11': 224, '12': 251}
  - test: total = 1700
    - {'5': 123, '6': 214, '7': 218, '8': 224, '9': 226, '10': 219, '11': 225, '12': 251}
- 2019:
  - train: total = 5556
    - {'1': 495, '2': 462, '3': 491, '4': 480, '5': 484, '6': 465, '7': 475, '8': 464, '9': 435, '10': 427, '11': 420, '12': 458}
  - val: total = 2777
    - {'1': 247, '2': 231, '3': 246, '4': 240, '5': 242, '6': 232, '7': 237, '8': 232, '9': 217, '10': 214, '11': 210, '12': 229}
  - test: total = 2783
    - {'1': 248, '2': 231, '3': 246, '4': 240, '5': 242, '6': 233, '7': 238, '8': 233, '9': 218, '10': 214, '11': 210, '12': 230}
- 2020
  - train: total = 5125,
    - {'1': 450, '2': 435, '3': 450, '4': 442, '5': 435, '6': 405, '7': 418, '8': 408, '9': 395, '10': 421, '11': 432, '12': 434}
  - val: total = 2563, 
    - {'1': 225, '2': 217, '3': 225, '4': 221, '5': 218, '6': 202, '7': 209, '8': 204, '9': 198, '10': 211, '11': 216, '12': 217}
  - test: total = 2567, 
    - {'1': 225, '2': 218, '3': 225, '4': 222, '5': 218, '6': 203, '7': 210, '8': 204, '9': 198, '10': 211, '11': 216, '12': 217}