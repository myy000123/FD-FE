"""
Training script for DLF
"""
from run import DLF_run
import multiprocessing

if __name__ == '__main__':
    # Windows 系统需要这个调用来支持多进程
    multiprocessing.freeze_support()
    
    DLF_run(model_name='DLF', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
             res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
