import enum


from energydiff.dataset.utils import NAME_SEASONS, TimeSeriesDataset, PIT, standard_normal_cdf, standard_normal_icdf
from energydiff.dataset.heat_pump import WPuQ
from energydiff.dataset.wpuq_trafo import WPuQTrafo
from energydiff.dataset.wpuq_pv import WPuQPV
from energydiff.dataset.lcl_electricity import LCLElectricityProfile
from energydiff.dataset.cossmic import CoSSMic

from energydiff.dataset.wpuq_pv import DIRECTION_CODE as WPUQ_PV_DIRECTION_CODE

# class DatasetType(enum.Enum):
#     HEAT_PUMP = enum.auto()
#     LCL_ELECTRICITY = enum.auto()

all_dataset = {
    'wpuq',
    'lcl_electricity',
    'cossmic',
    'wpuq_trafo',
    'wpuq_pv'
}

