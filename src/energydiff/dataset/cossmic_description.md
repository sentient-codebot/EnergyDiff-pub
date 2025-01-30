# CoSSMic Heat Pump Dataset

website: https://data.open-power-system-data.org/household_data/2020-04-15

paper: ...

format: csv

## Main Fields

original resolution: 1min

value type: cumulative energy consumption in kWh. 

* utc_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H:%M:%SZ
    - Description: Start of timeperiod in Coordinated Universal Time
* cet_cest_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%S%z
    - Description: Start of timeperiod in Central European (Summer-) Time
* interpolated
    - Type: string
    - Description: marker to indicate which columns are missing data in source data and has been interpolated (e.g. DE_KN_Residential1_grid_import;)
* DE_KN_{LOCATION}_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial warehouse building in kWh
* DE_KN_{LOCATION}_pv_1
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_{LOCATION}_pv_2
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_{LOCATION}_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building of a business in the crafts sector in kWh
* DE_KN_{LOCATION}_storage_charge
    - Type: number (float)
    - Description: Battery charging energy in a industrial building of a business in the crafts sector in kWh
* DE_KN_{LOCATION}_storage_decharge
    - Type: number (float)
    - Description: Energy in kWh
* DE_KN_{LOCATION}_area_offices
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_{LOCATION}_area_room_{ROOM_NUMBER}
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
- DE_KN_{LOCATION}_{APPLIANCE}
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh

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

Number of samples of of each task and year: (summarized from gpt, might be incorrect)

| Data Type     | Location Type | Year | Month   | Number of Samples |
|---------------|---------------|------|---------|-------------------|
| grid_import   | industrial    | 2015 | 1 to 10 | 0                 |
| grid_import   | industrial    | 2015 | 11      | 2                 |
| grid_import   | industrial    | 2015 | 12      | 31                |
| grid_import   | industrial    | 2015 | Total   | 33                |
| grid_import   | industrial    | 2016 | 1 to 12 | 1003              |
| grid_import   | industrial    | 2016 | Total   | 1003              |
| grid_import   | industrial    | 2017 | 1 to 6  | 399               |
| grid_import   | industrial    | 2017 | 7 to 9  | 92                |
| grid_import   | industrial    | 2017 | 10      | 11                |
| grid_import   | industrial    | 2017 | 11 to 12| 0                 |
| grid_import   | industrial    | 2017 | Total   | 594               |
| grid_import   | industrial    | 2018 | 1 to 12 | 0                 |
| grid_import   | industrial    | 2018 | Total   | 0                 |
| grid_import   | industrial    | 2019 | 1 to 12 | 0                 |
| grid_import   | industrial    | 2019 | Total   | 0                 |
| grid_import   | industrial    | Total |        | 1630              |
| grid_import   | residential   | 2015 | 1 to 3  | 0                 |
| grid_import   | residential   | 2015 | 4       | 15                |
| grid_import   | residential   | 2015 | 5 to 12 | 685               |
| grid_import   | residential   | 2015 | Total   | 700               |
| grid_import   | residential   | 2016 | 1 to 12 | 2137              |
| grid_import   | residential   | 2016 | Total   | 2137              |
| grid_import   | residential   | 2017 | 1 to 12 | 1384              |
| grid_import   | residential   | 2017 | Total   | 1384              |
| grid_import   | residential   | 2018 | 1 to 12 | 496               |
| grid_import   | residential   | 2018 | Total   | 496               |
| grid_import   | residential   | 2019 | 1 to 4  | 120               |
| grid_import   | residential   | 2019 | 5 to 12 | 0                 |
| grid_import   | residential   | 2019 | Total   | 120               |
| grid_import   | residential   | Total |        | 4837              |
| grid_import   | public        | 2015 | 1 to 12 | 0                 |
| grid_import   | public        | 2015 | Total   | 0                 |
| grid_import   | public        | 2016 | 1 to 4  | 0                 |
| grid_import   | public        | 2016 | 5 to 12 | 218               |
| grid_import   | public        | 2016 | Total   | 218               |
| grid_import   | public        | 2017 | 1       | 16                |
| grid_import   | public        | 2017 | 2 to 12 | 0                 |
| grid_import   | public        | 2017 | Total   | 16                |
| grid_import   | public        | 2018 | 1 to 12 | 0                 |
| grid_import   | public        | 2018 | Total   | 0                 |
| grid_import   | public        | 2019 | 1 to 12 | 0                 |
| grid_import   | public        | 2019 | Total   | 0                 |
| grid_import   | public        | Total |        | 234               |
| grid_import   | Total         |      |         | 6701              |
| pv            | industrial    | 2015 | 1 to 9  | 0                 |
| pv

            | industrial    | 2015 | 10 to 12| 215               |
| pv            | industrial    | 2015 | Total   | 215               |
| pv            | industrial    | 2016 | 1 to 12 | 1098              |
| pv            | industrial    | 2016 | Total   | 1098              |
| pv            | industrial    | 2017 | 1 to 10 | 471               |
| pv            | industrial    | 2017 | 11 to 12| 0                 |
| pv            | industrial    | 2017 | Total   | 502               |
| pv            | industrial    | 2018 | 1 to 12 | 0                 |
| pv            | industrial    | 2018 | Total   | 0                 |
| pv            | industrial    | 2019 | 1 to 12 | 0                 |
| pv            | industrial    | 2019 | Total   | 0                 |
| pv            | industrial    | Total |        | 1815              |
| pv            | residential   | 2015 | 1 to 4  | 0                 |
| pv            | residential   | 2015 | 5 to 12 | 374               |
| pv            | residential   | 2015 | Total   | 374               |
| pv            | residential   | 2016 | 1 to 12 | 1405              |
| pv            | residential   | 2016 | Total   | 1405              |
| pv            | residential   | 2017 | 1 to 12 | 1165              |
| pv            | residential   | 2017 | Total   | 1165              |
| pv            | residential   | 2018 | 1 to 8  | 367               |
| pv            | residential   | 2018 | 9 to 12 | 0                 |
| pv            | residential   | 2018 | Total   | 398               |
| pv            | residential   | 2019 | 1 to 12 | 0                 |
| pv            | residential   | 2019 | Total   | 0                 |
| pv            | residential   | Total |        | 3342              |
| pv            | public        | 2015 | 1 to 12 | 0                 |
| pv            | public        | 2015 | Total   | 0                 |
| pv            | public        | 2016 | 1 to 12 | 0                 |
| pv            | public        | 2016 | Total   | 0                 |
| pv            | public        | 2017 | 1 to 12 | 0                 |
| pv            | public        | 2017 | Total   | 0                 |
| pv            | public        | 2018 | 1 to 12 | 0                 |
| pv            | public        | 2018 | Total   | 0                 |
| pv            | public        | 2019 | 1 to 12 | 0                 |
| pv            | public        | 2019 | Total   | 0                 |
| pv            | public        | Total |        | 0                 |
| pv            | Total         |      |         | 5157              |


## Appendix

### Train/val/test Original Statistics

Statistics:
grid_import_industrial_2015_1: 0
grid_import_industrial_2015_2: 0
grid_import_industrial_2015_3: 0
grid_import_industrial_2015_4: 0
grid_import_industrial_2015_5: 0
grid_import_industrial_2015_6: 0
grid_import_industrial_2015_7: 0
grid_import_industrial_2015_8: 0
grid_import_industrial_2015_9: 0
grid_import_industrial_2015_10: 0
grid_import_industrial_2015_11: 2
grid_import_industrial_2015_12: 31
grid_import_industrial_2015: 33
grid_import_industrial_2016_1: 31
grid_import_industrial_2016_2: 54
grid_import_industrial_2016_3: 93
grid_import_industrial_2016_4: 90
grid_import_industrial_2016_5: 93
grid_import_industrial_2016_6: 90
grid_import_industrial_2016_7: 93
grid_import_industrial_2016_8: 93
grid_import_industrial_2016_9: 90
grid_import_industrial_2016_10: 93
grid_import_industrial_2016_11: 90
grid_import_industrial_2016_12: 93
grid_import_industrial_2016: 1003
grid_import_industrial_2017_1: 93
grid_import_industrial_2017_2: 84
grid_import_industrial_2017_3: 93
grid_import_industrial_2017_4: 90
grid_import_industrial_2017_5: 93
grid_import_industrial_2017_6: 38
grid_import_industrial_2017_7: 31
grid_import_industrial_2017_8: 31
grid_import_industrial_2017_9: 30
grid_import_industrial_2017_10: 11
grid_import_industrial_2017_11: 0
grid_import_industrial_2017_12: 0
grid_import_industrial_2017: 594
grid_import_industrial_2018_1: 0
grid_import_industrial_2018_2: 0
grid_import_industrial_2018_3: 0
grid_import_industrial_2018_4: 0
grid_import_industrial_2018_5: 0
grid_import_industrial_2018_6: 0
grid_import_industrial_2018_7: 0
grid_import_industrial_2018_8: 0
grid_import_industrial_2018_9: 0
grid_import_industrial_2018_10: 0
grid_import_industrial_2018_11: 0
grid_import_industrial_2018_12: 0
grid_import_industrial_2018: 0
grid_import_industrial_2019_1: 0
grid_import_industrial_2019_2: 0
grid_import_industrial_2019_3: 0
grid_import_industrial_2019_4: 0
grid_import_industrial_2019_5: 0
grid_import_industrial_2019_6: 0
grid_import_industrial_2019_7: 0
grid_import_industrial_2019_8: 0
grid_import_industrial_2019_9: 0
grid_import_industrial_2019_10: 0
grid_import_industrial_2019_11: 0
grid_import_industrial_2019_12: 0
grid_import_industrial_2019: 0
grid_import_industrial: 1630
grid_import_residential_2015_1: 0
grid_import_residential_2015_2: 0
grid_import_residential_2015_3: 0
grid_import_residential_2015_4: 15
grid_import_residential_2015_5: 41
grid_import_residential_2015_6: 60
grid_import_residential_2015_7: 62
grid_import_residential_2015_8: 62
grid_import_residential_2015_9: 60
grid_import_residential_2015_10: 95
grid_import_residential_2015_11: 150
grid_import_residential_2015_12: 155
grid_import_residential_2015: 700
grid_import_residential_2016_1: 155
grid_import_residential_2016_2: 146
grid_import_residential_2016_3: 186
grid_import_residential_2016_4: 180
grid_import_residential_2016_5: 186
grid_import_residential_2016_6: 180
grid_import_residential_2016_7: 186
grid_import_residential_2016_8: 186
grid_import_residential_2016_9: 180
grid_import_residential_2016_10: 186
grid_import_residential_2016_11: 180
grid_import_residential_2016_12: 186
grid_import_residential_2016: 2137
grid_import_residential_2017_1: 186
grid_import_residential_2017_2: 140
grid_import_residential_2017_3: 135
grid_import_residential_2017_4: 120
grid_import_residential_2017_5: 124
grid_import_residential_2017_6: 120
grid_import_residential_2017_7: 100
grid_import_residential_2017_8: 93
grid_import_residential_2017_9: 90
grid_import_residential_2017_10: 93
grid_import_residential_2017_11: 90
grid_import_residential_2017_12: 93
grid_import_residential_2017: 1384
grid_import_residential_2018_1: 93
grid_import_residential_2018_2: 59
grid_import_residential_2018_3: 62
grid_import_residential_2018_4: 37
grid_import_residential_2018_5: 31
grid_import_residential_2018_6: 30
grid_import_residential_2018_7: 31
grid_import_residential_2018_8: 31
grid_import_residential_2018_9: 30
grid_import_residential_2018_10: 31
grid_import_residential_2018_11: 30
grid_import_residential_2018_12: 31
grid_import_residential_2018: 496
grid_import_residential_2019_1: 31
grid_import_residential_2019_2: 28
grid_import_residential_2019_3: 31
grid_import_residential_2019_4: 30
grid_import_residential_2019_5: 0
grid_import_residential_2019_6: 0
grid_import_residential_2019_7: 0
grid_import_residential_2019_8: 0
grid_import_residential_2019_9: 0
grid_import_residential_2019_10: 0
grid_import_residential_2019_11: 0
grid_import_residential_2019_12: 0
grid_import_residential_2019: 120
grid_import_residential: 4837
grid_import_public_2015_1: 0
grid_import_public_2015_2: 0
grid_import_public_2015_3: 0
grid_import_public_2015_4: 0
grid_import_public_2015_5: 0
grid_import_public_2015_6: 0
grid_import_public_2015_7: 0
grid_import_public_2015_8: 0
grid_import_public_2015_9: 0
grid_import_public_2015_10: 0
grid_import_public_2015_11: 0
grid_import_public_2015_12: 0
grid_import_public_2015: 0
grid_import_public_2016_1: 0
grid_import_public_2016_2: 0
grid_import_public_2016_3: 0
grid_import_public_2016_4: 0
grid_import_public_2016_5: 14
grid_import_public_2016_6: 30
grid_import_public_2016_7: 31
grid_import_public_2016_8: 31
grid_import_public_2016_9: 30
grid_import_public_2016_10: 31
grid_import_public_2016_11: 21
grid_import_public_2016_12: 30
grid_import_public_2016: 218
grid_import_public_2017_1: 16
grid_import_public_2017_2: 0
grid_import_public_2017_3: 0
grid_import_public_2017_4: 0
grid_import_public_2017_5: 0
grid_import_public_2017_6: 0
grid_import_public_2017_7: 0
grid_import_public_2017_8: 0
grid_import_public_2017_9: 0
grid_import_public_2017_10: 0
grid_import_public_2017_11: 0
grid_import_public_2017_12: 0
grid_import_public_2017: 16
grid_import_public_2018_1: 0
grid_import_public_2018_2: 0
grid_import_public_2018_3: 0
grid_import_public_2018_4: 0
grid_import_public_2018_5: 0
grid_import_public_2018_6: 0
grid_import_public_2018_7: 0
grid_import_public_2018_8: 0
grid_import_public_2018_9: 0
grid_import_public_2018_10: 0
grid_import_public_2018_11: 0
grid_import_public_2018_12: 0
grid_import_public_2018: 0
grid_import_public_2019_1: 0
grid_import_public_2019_2: 0
grid_import_public_2019_3: 0
grid_import_public_2019_4: 0
grid_import_public_2019_5: 0
grid_import_public_2019_6: 0
grid_import_public_2019_7: 0
grid_import_public_2019_8: 0
grid_import_public_2019_9: 0
grid_import_public_2019_10: 0
grid_import_public_2019_11: 0
grid_import_public_2019_12: 0
grid_import_public_2019: 0
grid_import_public: 234
grid_import: 6701
pv_industrial_2015_1: 0
pv_industrial_2015_2: 0
pv_industrial_2015_3: 0
pv_industrial_2015_4: 0
pv_industrial_2015_5: 0
pv_industrial_2015_6: 0
pv_industrial_2015_7: 0
pv_industrial_2015_8: 0
pv_industrial_2015_9: 0
pv_industrial_2015_10: 32
pv_industrial_2015_11: 90
pv_industrial_2015_12: 93
pv_industrial_2015: 215
pv_industrial_2016_1: 93
pv_industrial_2016_2: 87
pv_industrial_2016_3: 93
pv_industrial_2016_4: 90
pv_industrial_2016_5: 93
pv_industrial_2016_6: 90
pv_industrial_2016_7: 93
pv_industrial_2016_8: 93
pv_industrial_2016_9: 90
pv_industrial_2016_10: 93
pv_industrial_2016_11: 90
pv_industrial_2016_12: 93
pv_industrial_2016: 1098
pv_industrial_2017_1: 93
pv_industrial_2017_2: 84
pv_industrial_2017_3: 67
pv_industrial_2017_4: 60
pv_industrial_2017_5: 62
pv_industrial_2017_6: 33
pv_industrial_2017_7: 31
pv_industrial_2017_8: 31
pv_industrial_2017_9: 30
pv_industrial_2017_10: 11
pv_industrial_2017_11: 0
pv_industrial_2017_12: 0
pv_industrial_2017: 502
pv_industrial_2018_1: 0
pv_industrial_2018_2: 0
pv_industrial_2018_3: 0
pv_industrial_2018_4: 0
pv_industrial_2018_5: 0
pv_industrial_2018_6: 0
pv_industrial_2018_7: 0
pv_industrial_2018_8: 0
pv_industrial_2018_9: 0
pv_industrial_2018_10: 0
pv_industrial_2018_11: 0
pv_industrial_2018_12: 0
pv_industrial_2018: 0
pv_industrial_2019_1: 0
pv_industrial_2019_2: 0
pv_industrial_2019_3: 0
pv_industrial_2019_4: 0
pv_industrial_2019_5: 0
pv_industrial_2019_6: 0
pv_industrial_2019_7: 0
pv_industrial_2019_8: 0
pv_industrial_2019_9: 0
pv_industrial_2019_10: 0
pv_industrial_2019_11: 0
pv_industrial_2019_12: 0
pv_industrial_2019: 0
pv_industrial: 1815
pv_residential_2015_1: 0
pv_residential_2015_2: 0
pv_residential_2015_3: 0
pv_residential_2015_4: 0
pv_residential_2015_5: 10
pv_residential_2015_6: 30
pv_residential_2015_7: 31
pv_residential_2015_8: 31
pv_residential_2015_9: 30
pv_residential_2015_10: 59
pv_residential_2015_11: 90
pv_residential_2015_12: 93
pv_residential_2015: 374
pv_residential_2016_1: 93
pv_residential_2016_2: 88
pv_residential_2016_3: 124
pv_residential_2016_4: 120
pv_residential_2016_5: 124
pv_residential_2016_6: 120
pv_residential_2016_7: 124
pv_residential_2016_8: 124
pv_residential_2016_9: 120
pv_residential_2016_10: 124
pv_residential_2016_11: 120
pv_residential_2016_12: 124
pv_residential_2016: 1405
pv_residential_2017_1: 124
pv_residential_2017_2: 112
pv_residential_2017_3: 104
pv_residential_2017_4: 90
pv_residential_2017_5: 93
pv_residential_2017_6: 90
pv_residential_2017_7: 93
pv_residential_2017_8: 93
pv_residential_2017_9: 90
pv_residential_2017_10: 93
pv_residential_2017_11: 90
pv_residential_2017_12: 93
pv_residential_2017: 1165
pv_residential_2018_1: 93
pv_residential_2018_2: 59
pv_residential_2018_3: 62
pv_residential_2018_4: 37
pv_residential_2018_5: 31
pv_residential_2018_6: 30
pv_residential_2018_7: 31
pv_residential_2018_8: 31
pv_residential_2018_9: 24
pv_residential_2018_10: 0
pv_residential_2018_11: 0
pv_residential_2018_12: 0
pv_residential_2018: 398
pv_residential_2019_1: 0
pv_residential_2019_2: 0
pv_residential_2019_3: 0
pv_residential_2019_4: 0
pv_residential_2019_5: 0
pv_residential_2019_6: 0
pv_residential_2019_7: 0
pv_residential_2019_8: 0
pv_residential_2019_9: 0
pv_residential_2019_10: 0
pv_residential_2019_11: 0
pv_residential_2019_12: 0
pv_residential_2019: 0
pv_residential: 3342
pv_public_2015_1: 0
pv_public_2015_2: 0
pv_public_2015_3: 0
pv_public_2015_4: 0
pv_public_2015_5: 0
pv_public_2015_6: 0
pv_public_2015_7: 0
pv_public_2015_8: 0
pv_public_2015_9: 0
pv_public_2015_10: 0
pv_public_2015_11: 0
pv_public_2015_12: 0
pv_public_2015: 0
pv_public_2016_1: 0
pv_public_2016_2: 0
pv_public_2016_3: 0
pv_public_2016_4: 0
pv_public_2016_5: 0
pv_public_2016_6: 0
pv_public_2016_7: 0
pv_public_2016_8: 0
pv_public_2016_9: 0
pv_public_2016_10: 0
pv_public_2016_11: 0
pv_public_2016_12: 0
pv_public_2016: 0
pv_public_2017_1: 0
pv_public_2017_2: 0
pv_public_2017_3: 0
pv_public_2017_4: 0
pv_public_2017_5: 0
pv_public_2017_6: 0
pv_public_2017_7: 0
pv_public_2017_8: 0
pv_public_2017_9: 0
pv_public_2017_10: 0
pv_public_2017_11: 0
pv_public_2017_12: 0
pv_public_2017: 0
pv_public_2018_1: 0
pv_public_2018_2: 0
pv_public_2018_3: 0
pv_public_2018_4: 0
pv_public_2018_5: 0
pv_public_2018_6: 0
pv_public_2018_7: 0
pv_public_2018_8: 0
pv_public_2018_9: 0
pv_public_2018_10: 0
pv_public_2018_11: 0
pv_public_2018_12: 0
pv_public_2018: 0
pv_public_2019_1: 0
pv_public_2019_2: 0
pv_public_2019_3: 0
pv_public_2019_4: 0
pv_public_2019_5: 0
pv_public_2019_6: 0
pv_public_2019_7: 0
pv_public_2019_8: 0
pv_public_2019_9: 0
pv_public_2019_10: 0
pv_public_2019_11: 0
pv_public_2019_12: 0
pv_public_2019: 0
pv_public: 0
pv: 5157

### All Fields in Original Dataset

household_data_1min_singleindex.csv
---

* utc_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%SZ
    - Description: Start of timeperiod in Coordinated Universal Time
* cet_cest_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%S%z
    - Description: Start of timeperiod in Central European (Summer-) Time
* interpolated
    - Type: string
    - Description: marker to indicate which columns are missing data in source data and has been interpolated (e.g. DE_KN_Residential1_grid_import;)
* DE_KN_industrial1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_1
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_2
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_charge
    - Type: number (float)
    - Description: Battery charging energy in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_decharge
    - Type: number (float)
    - Description: Energy in kWh
* DE_KN_industrial3_area_offices
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_1
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_2
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_3
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_4
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_compressor
    - Type: number (float)
    - Description: Compressor energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_aggregate
    - Type: number (float)
    - Description: Cooling aggregate energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_pumps
    - Type: number (float)
    - Description: Cooling pumps energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_1
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_2
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_3
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_4
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_5
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_facade
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_roof
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ventilation
    - Type: number (float)
    - Description: Ventilation energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_public1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_public2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_residential1_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential1_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the suburban area in kWh
* DE_KN_residential1_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential2_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential3_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a residential building, located in the urban area in kWh
* DE_KN_residential4_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential4_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential5_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential6_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential6_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh

