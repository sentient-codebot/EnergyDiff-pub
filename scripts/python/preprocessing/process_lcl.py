from energydiff.dataset import LCLElectricityProfile

lcl_data = LCLElectricityProfile(
    root = 'data/',
    load = True,
    conditioning = False,
    normalize = True,
    pit_transform = False,
    shuffle = True,
    vectorize = True,
    style_vectorize = 'chronological',
    vectorize_window_size = 3,
)