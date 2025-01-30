from datetime import datetime
import random

__all__ = ['io', 'plot', 'eval', 'generic','configuration', 'sample', 'argument_parser']

def generate_time_id():
    return datetime.now().strftime("%Y%m%d" + "-" + f"{random.randint(0, 9999):04d}")