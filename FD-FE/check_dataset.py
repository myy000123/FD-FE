#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看数据集详细信息
"""

import pickle
import numpy as np

def check_dataset_info(pkl_path, dataset_name):
    print(f"\n=== {dataset_name} 数据集信息 ===")
    print(f"文件路径: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n数据集包含的模式: {list(data.keys())}")
    
    total_samples = 0
    for mode in ['train', 'valid', 'test']:
        if mode in data:
            print(f"\n--- {mode.upper()} 集 ---")
            mode_data = data[mode]
            print(f"包含的数据类型: {list(mode_data.keys())}")
            
            for key, value in mode_data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                    if key == 'regression_labels':
                        sample_count = len(value)
                        total_samples += sample_count
                        print(f"    样本数量: {sample_count}")
                        print(f"    标签范围: [{np.min(value):.3f}, {np.max(value):.3f}]")
                        print(f"    标签均值: {np.mean(value):.3f}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: 列表长度 {len(value)}")
                    if len(value) > 0:
                        print(f"    第一个元素类型: {type(value[0])}")
                else:
                    print(f"  {key}: {type(value)}")
    
    print(f"\n总样本数量: {total_samples}")
    return total_samples

def main():
    # 检查MOSI数据集
    mosi_total = check_dataset_info('./dataset/MOSI/Processed/aligned_50.pkl', 'MOSI')
    
    # 检查MOSEI数据集（如果存在）
    try:
        mosei_total = check_dataset_info('./dataset/MOSEI/Processed/aligned_50.pkl', 'MOSEI')
    except FileNotFoundError:
        print("\n=== MOSEI 数据集 ===")
        print("文件不存在: ./dataset/MOSEI/Processed/aligned_50.pkl")
        mosei_total = 0
    
    print("\n=== 数据集概览 ===")
    print(f"MOSI总样本数: {mosi_total}")
    print(f"MOSEI总样本数: {mosei_total}")

if __name__ == '__main__':
    main() 