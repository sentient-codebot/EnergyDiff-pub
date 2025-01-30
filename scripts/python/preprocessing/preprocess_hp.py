from argparse import ArgumentParser

from energydiff.dataset.heat_pump import PreHeatPump

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--year', type=int, default=2018)
    
    args = arg_parser.parse_args()
    
    prehp = PreHeatPump(
        root='data/raw/',
        year=args.year,
    )

    dicts_num_samples = prehp.load_process_save(num_process=4)
    name_task = ['train', 'val', 'test']
    for idx, dict in enumerate(dicts_num_samples):
        print(f'Task: {name_task[idx]}:')
        print(dict)
        print(f'Total: {sum(dict.values())}')
        