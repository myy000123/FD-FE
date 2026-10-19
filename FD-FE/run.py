import gc
import logging
import os
import time
from pathlib import Path, PosixPath, WindowsPath
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import DLF
from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
from trains.singleTask.misc import softmax
import sys
import platform
import pickle

from datetime import datetime      
now = datetime.now()
format = "%Y/%m/%d %H:%M:%S"
formatted_now = now.strftime(format)
formatted_now = str(formatted_now)+" - "

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DLF_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=0, verbose_level=1, mode = '', is_training = False 
):
    # Initialization
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_training = is_training  
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_training:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time"]+["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        
        res = [formatted_now]+res 
        df.loc[len(df)] = res    
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False): 
    # Windows 系统建议使用较少的 workers 避免问题
    import platform
    if platform.system() == 'Windows':
        num_workers = min(num_workers, 2)
    
    dataloader = MMDataLoader(args, num_workers)

    if args.is_training:
        print("training for DLF")

        
        args.gd_size_low = 64  
        args.w_losses_low = [1, 10]  
        args.metric_low = 'l1'  

        
        args.gd_size_high = 32  
        args.w_losses_high = [1, 10]  
        args.metric_high = 'l1'  

        to_idx = [0, 1, 2]  
        from_idx = [0, 1, 2]  
        assert len(from_idx) >= 1

        model = []
        model_DLF = getattr(DLF, 'DLF')(args)

        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(n_classes=1,
                                                                               hidden_size=
                                                                               args.dst_feature_dim_nheads[0],
                                                                               gd_size=args.gd_size_low,
                                                                               to_idx=to_idx, from_idx=from_idx,
                                                                               gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
                                                                               gd_reg=10,
                                                                               w_losses=args.w_losses_low,
                                                                               metric=args.metric_low,
                                                                               alpha=1 / 8,
                                                                               hyp_params=args)

        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(n_classes=1,
                                                                                   hidden_size=
                                                                                   args.dst_feature_dim_nheads[0] * 2,
                                                                                   gd_size=args.gd_size_high,
                                                                                   to_idx=to_idx, from_idx=from_idx,
                                                                                   gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
                                                                                   gd_reg=10,
                                                                                   w_losses=args.w_losses_high,
                                                                                   metric=args.metric_high,
                                                                                   alpha=1 / 8,
                                                                                   hyp_params=args)

        model_DLF = model_DLF.cuda()

        model = [model_DLF]         
    else:
        print("testing phase for DLF")
        model = getattr(DLF, 'DLF')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)

    def load_model_with_path_fix(model_path, device=None):
        """
        安全加载模型，处理跨平台路径兼容性问题
        """
        if device is None:
            device = args.device
        
        # 首先尝试使用 weights_only=True（推荐方式）
        try:
            return torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e1:
            print(f"尝试 weights_only=True 加载失败: {e1}")
            
            # 如果失败，尝试设置路径兼容性
            try:
                # 临时修复 pathlib 的兼容性问题
                import pathlib
                temp_posix_path = pathlib.PosixPath
                if platform.system() == 'Windows':
                    pathlib.PosixPath = pathlib.WindowsPath
                
                try:
                    result = torch.load(model_path, map_location=device)
                    return result
                finally:
                    # 恢复原始的 PosixPath
                    pathlib.PosixPath = temp_posix_path
                    
            except Exception as e2:
                print(f"尝试路径修复加载失败: {e2}")
                
                # 最后尝试标准方式但映射到CPU再转移
                try:
                    return torch.load(model_path, map_location='cpu')
                except Exception as e3:
                    print(f"所有加载方式都失败了: {e3}")
                    raise e3

    #test
    if args.mode == 'test':
        model_path = './pt/DLF'+str(args.dataset_name)+'.pth'
        try:
            state_dict = load_model_with_path_fix(model_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载模型文件: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保模型文件存在且格式正确")
            return {}
            
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    #train
    else:
        '''epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load('./pt/DLF'+str(args.dataset_name)+'.pth'))'''
        #model.load_state_dict(torch.load('./pt/DLF'+str(args.dataset_name)+'.pth'))
        
        pretrained_path = '/home/cqupt/MYY/DLF-main-ronghe/pt/DLF'+str(args.dataset_name)+'.pth'
        if os.path.exists(pretrained_path):
        # 加载保存的配置
          saved_data = torch.load(pretrained_path, map_location=args.device)
          saved_args = saved_data['args']
          model[0].load_state_dict(saved_data['model_state_dict'])
          print(f"Loaded pretrained weights from {pretrained_path}")
          

        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results