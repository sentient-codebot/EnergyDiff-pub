from energydiff.dataset.cossmic import PreCoSSMic, CoSSMic

if __name__ == '__main__':
    # prehp = PreCoSSMic(
    #     root='data/cossmic/',
    #     load_pickle=True,
    # )
    
    cossmic_data = CoSSMic(
        root='data/cossmic/',
    )
    print(cossmic_data)
    pass