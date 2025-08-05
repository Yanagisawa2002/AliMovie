#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验启动脚本
提供便捷的实验配置和启动方式

使用方法:
python run_experiment.py --preset quick_demo
python run_experiment.py --preset large_model
python run_experiment.py --custom
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

def load_config(config_path: str = "config_example.json"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def create_experiment_command(preset_name: str = None, custom_args: dict = None):
    """创建实验命令"""
    
    if preset_name:
        # 使用预设配置
        config = load_config()
        if not config or preset_name not in config.get('presets', {}):
            print(f"❌ 预设配置不存在: {preset_name}")
            return None
        
        preset = config['presets'][preset_name]
        
        # 构建命令参数
        cmd_args = ['python', 'main.py', '--mode', 'full']
        
        # 添加模型参数
        if 'model_config' in preset:
            model_config = preset['model_config']
            for key, value in model_config.items():
                cmd_args.extend([f'--{key}', str(value)])
        
        # 添加训练参数
        if 'training_config' in preset:
            training_config = preset['training_config']
            for key, value in training_config.items():
                cmd_args.extend([f'--{key}', str(value)])
        
        # 添加数据参数
        if 'data_config' in preset:
            data_config = preset['data_config']
            for key, value in data_config.items():
                if key != 'data_path':  # data_path通过--data_path传递
                    cmd_args.extend([f'--{key}', str(value)])
        
        return cmd_args
    
    elif custom_args:
        # 使用自定义参数
        cmd_args = ['python', 'main.py']
        
        for key, value in custom_args.items():
            cmd_args.extend([f'--{key}', str(value)])
        
        return cmd_args
    
    else:
        # 默认配置
        return ['python', 'main.py', '--mode', 'full']

def run_experiment(cmd_args: list, experiment_name: str = None):
    """运行实验"""
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🚀 启动实验: {experiment_name}")
    print(f"📝 命令: {' '.join(cmd_args)}")
    print("" + "="*60)
    
    try:
        # 运行命令
        result = subprocess.run(cmd_args, check=True, capture_output=False)
        
        print("" + "="*60)
        print(f"✅ 实验 {experiment_name} 完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print("" + "="*60)
        print(f"❌ 实验 {experiment_name} 失败")
        print(f"错误代码: {e.returncode}")
        return False
    
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
        return False

def interactive_config():
    """交互式配置"""
    print("🔧 交互式实验配置")
    print("" + "="*40)
    
    config = {}
    
    # 基本配置
    print("\n📊 基本配置:")
    config['mode'] = input("运行模式 (train/evaluate/full) [full]: ") or 'full'
    
    # 模型配置
    print("\n🤖 模型配置:")
    config['d_model'] = int(input("模型维度 [256]: ") or 256)
    config['n_heads'] = int(input("注意力头数 [8]: ") or 8)
    config['n_layers'] = int(input("Transformer层数 [6]: ") or 6)
    config['max_seq_len'] = int(input("最大序列长度 [50]: ") or 50)
    
    # 训练配置
    print("\n🔥 训练配置:")
    config['pretrain_epochs'] = int(input("预训练轮数 [1]: ") or 1)
    config['finetune_epochs'] = int(input("微调轮数 [2]: ") or 2)
    config['batch_size'] = int(input("批次大小 [64]: ") or 64)
    
    # 数据配置
    print("\n📦 数据配置:")
    config['long_tail_threshold'] = int(input("长尾用户阈值 [5]: ") or 5)
    config['pretrain_sample_frac'] = float(input("预训练采样比例 [0.1]: ") or 0.1)
    
    return config

def show_presets():
    """显示可用的预设配置"""
    config = load_config()
    if not config:
        return
    
    presets = config.get('presets', {})
    
    print("📋 可用的预设配置:")
    print("" + "="*40)
    
    for name, preset in presets.items():
        if name.startswith('_'):
            continue
            
        print(f"\n🎯 {name}:")
        
        if 'model_config' in preset:
            model = preset['model_config']
            print(f"  模型: d_model={model.get('d_model', 'N/A')}, "
                  f"n_layers={model.get('n_layers', 'N/A')}, "
                  f"n_heads={model.get('n_heads', 'N/A')}")
        
        if 'training_config' in preset:
            training = preset['training_config']
            print(f"  训练: pretrain_epochs={training.get('pretrain_epochs', 'N/A')}, "
                  f"finetune_epochs={training.get('finetune_epochs', 'N/A')}, "
                  f"batch_size={training.get('batch_size', 'N/A')}")
    
    print("\n使用方法: python run_experiment.py --preset <preset_name>")

def main():
    parser = argparse.ArgumentParser(description='实验启动脚本')
    
    parser.add_argument('--preset', type=str, help='使用预设配置')
    parser.add_argument('--interactive', action='store_true', help='交互式配置')
    parser.add_argument('--list-presets', action='store_true', help='显示可用预设')
    parser.add_argument('--name', type=str, help='实验名称')
    
    # 允许传递自定义参数
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--pretrain_epochs', type=int)
    parser.add_argument('--finetune_epochs', type=int)
    
    args = parser.parse_args()
    
    # 显示预设配置
    if args.list_presets:
        show_presets()
        return
    
    # 交互式配置
    if args.interactive:
        custom_config = interactive_config()
        cmd_args = create_experiment_command(custom_args=custom_config)
        
        if cmd_args:
            run_experiment(cmd_args, args.name)
        return
    
    # 使用预设配置
    if args.preset:
        cmd_args = create_experiment_command(preset_name=args.preset)
        
        if cmd_args:
            experiment_name = args.name or f"{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_experiment(cmd_args, experiment_name)
        return
    
    # 使用命令行参数
    custom_args = {}
    for key, value in vars(args).items():
        if value is not None and key not in ['preset', 'interactive', 'list_presets', 'name']:
            custom_args[key] = value
    
    if custom_args:
        cmd_args = create_experiment_command(custom_args=custom_args)
    else:
        cmd_args = create_experiment_command()
    
    if cmd_args:
        run_experiment(cmd_args, args.name)

if __name__ == "__main__":
    print("🧪 推荐系统实验启动器")
    print("" + "="*50)
    
    # 检查依赖
    if not os.path.exists('main.py'):
        print("❌ 未找到 main.py，请确保在正确的目录中运行")
        sys.exit(1)
    
    main()