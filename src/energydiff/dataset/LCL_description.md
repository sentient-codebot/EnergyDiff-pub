# Low Carbon London Dataset

website: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

paper: https://www.researchgate.net/profile/James-Schofield-8/publication/293176172_Low_Carbon_London_project_Data_from_the_dynamic_time-of-use_electricity_pricing_trial_2013/links/56b6889d08ae5ad36059b61c/Low-Carbon-London-project-Data-from-the-dynamic-time-of-use-electricity-pricing-trial-2013.pdf

project information: https://innovation.ukpowernetworks.co.uk/projects/low-carbon-london

format: csv

unit: kWh

resolution: 30min 

group:
- group D: using dynamic time-of-use (dToU) tariffs in 2013
- group N: remained on the existing non-dynamic tariffs

Ensured that two groups are approximately representative of London. 

dynamic tariffs: three levels, default, high and low. Informed one day in advance. 
price changes are random due to system balancing and distribution network constraint management. 
More details are in the document (paper). 

The original datasets consist of 168 .csv files (Nr. 0~167) of similar sizes. 

ToU tariff profiles exist in files
- 134-167

## Usage

Already pre-processed to
- remove missing values
- reshaped to weekly data, i.e. [samples, $7*24*2$]
- saved to tensor in `$dataroot$/raw/*.pt` files. 

### Train/val/test Partitioning

**Unconditional generation + standard tariff.** 
The first 0-133 files contain only standard tariff data. 
- use file 0-99 for training ~ 2,061,748 person-days
- use file 100-116 for validation ~ 350,817 person-days
- use file 117-133 for testing ~ 351,115 person-days

**by default use 1%**

File names
- 'lcl_electricity_train.pt'
- 'lcl_electricity_val.pt'
- 'lcl_electricity_test.pt'

*Below are planned data preprecessing plan.*

**Conditional generation (week of the year + tariff type)**

**Conditional generation (with tariff time series)**