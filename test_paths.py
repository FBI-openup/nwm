#!/usr/bin/env python3
"""
路径验证测试脚本
验证项目重构后所有导入路径和文件路径是否正确
"""

import os
import sys
import importlib.util
from pathlib import Path

# 确保当前脚本所在目录在Python路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def test_file_exists(file_path, description):
    """测试文件是否存在"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - 文件不存在")
        return False

def test_import(module_name, description):
    """测试模块导入是否成功"""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ {description}: {module_name} - 其他错误: {e}")
        return False

def test_scripts_import(script_name, description):
    """测试scripts目录下的模块导入"""
    try:
        module_name = f"scripts.{script_name}"
        exec(f"from scripts import {script_name}")
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: scripts.{script_name} - 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ {description}: scripts.{script_name} - 其他错误: {e}")
        return False

def main():
    print("🔍 开始验证项目路径...")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # 1. 验证核心Python模块
    print("\n📦 验证核心模块导入:")
    core_modules = [
        ("datasets", "数据集模块"),
        ("models", "模型模块"),
        ("misc", "工具模块"),
        ("diffusion", "扩散模块"),
        ("hybrid_models", "混合模型"),
        ("distributed", "分布式模块"),
        ("latent_dataset", "潜在数据集"),
    ]
    
    for module, desc in core_modules:
        total += 1
        if test_import(module, desc):
            passed += 1
    
    # 2. 验证scripts目录模块
    print("\n📜 验证scripts目录模块:")
    script_modules = [
        ("isolated_nwm_infer", "隔离推理模块"),
        ("isolated_nwm_eval", "隔离评估模块"),
        ("planning_eval", "规划评估模块"),
        ("encode_latents", "潜在编码模块"),
        ("worldmem_setup_and_test", "WorldMem设置测试"),
    ]
    
    for script, desc in script_modules:
        total += 1
        if test_scripts_import(script, desc):
            passed += 1
    
    # 3. 验证配置文件
    print("\n⚙️ 验证配置文件:")
    config_files = [
        ("config/data_config.yaml", "数据配置"),
        ("config/eval_config.yaml", "评估配置"),
        ("config/memory_config.yaml", "内存配置"),
        ("config/nwm_cdit_l_latents_L40S.yaml", "L40S配置"),
    ]
    
    for file_path, desc in config_files:
        total += 1
        if test_file_exists(file_path, desc):
            passed += 1
    
    # 4. 验证脚本文件
    print("\n🔧 验证脚本文件:")
    script_files = [
        ("scripts/setup_nwm_env.sh", "环境设置脚本"),
        ("scripts/train_L40S_slurm.sh", "SLURM训练脚本"),
        ("scripts/encode_all_datasets.sh", "数据集编码脚本"),
    ]
    
    for file_path, desc in script_files:
        total += 1
        if test_file_exists(file_path, desc):
            passed += 1
    
    # 5. 验证文档文件
    print("\n📚 验证文档文件:")
    doc_files = [
        ("docs/README_Hybrid_Memory.md", "混合内存文档"),
        ("docs/README_Latent_Encoding.md", "潜在编码文档"),
        ("docs/WorldMem_Setup_README.md", "WorldMem设置文档"),
        ("README.md", "主文档"),
    ]
    
    for file_path, desc in doc_files:
        total += 1
        if test_file_exists(file_path, desc):
            passed += 1
    
    # 6. 验证目录结构
    print("\n📁 验证目录结构:")
    directories = [
        ("scripts", "脚本目录"),
        ("docs", "文档目录"),
        ("config", "配置目录"),
        ("diffusion", "扩散模块目录"),
        ("data", "数据目录"),
        ("logs", "日志目录"),
    ]
    
    for dir_path, desc in directories:
        total += 1
        if test_file_exists(dir_path, desc):
            passed += 1
    
    # 测试结果
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有路径验证通过！项目重构成功。")
        return True
    else:
        failed = total - passed
        print(f"⚠️ {failed} 个路径验证失败，请检查上述错误。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
