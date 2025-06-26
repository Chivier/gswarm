#!/usr/bin/env python3
"""
JSON模型配置修正脚本
确保所有使用base_model的模型都包含完整的参数（包括memory_gb等）
"""

import json
import os
import glob
import copy
from typing import Dict, Any

def load_json_file(filepath: str) -> Dict[Any, Any]:
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return {}

def save_json_file(filepath: str, data: Dict[Any, Any]) -> bool:
    """保存JSON文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error saving {filepath}: {e}")
        return False

def expand_model_inheritance(models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    展开模型继承关系，为所有模型添加完整参数
    """
    result = copy.deepcopy(models)
    
    # 多次迭代处理可能的链式继承
    max_iterations = 5
    for iteration in range(max_iterations):
        changes_made = False
        
        for model_id, model_config in result.items():
            if 'base_model' in model_config:
                base_model_id = model_config['base_model']
                
                if base_model_id not in result:
                    print(f"⚠️  Warning: Base model '{base_model_id}' not found for '{model_id}'")
                    continue
                
                base_config = result[base_model_id]
                
                # 从base_model继承所有缺失的参数
                for key, value in base_config.items():
                    if key not in model_config:
                        model_config[key] = copy.deepcopy(value)
                        changes_made = True
                        print(f"  📝 Added '{key}': {value} to '{model_id}'")
        
        if not changes_made:
            break
    
    return result

def process_config_file(filepath: str, dry_run: bool = False) -> bool:
    """处理单个配置文件"""
    print(f"\n🔍 Processing: {filepath}")
    
    # 加载文件
    data = load_json_file(filepath)
    if not data:
        return False
    
    # 检查是否有models部分
    if 'models' not in data:
        print(f"  ℹ️  No 'models' section found, skipping...")
        return True
    
    # 处理模型继承
    original_models = data['models']
    expanded_models = expand_model_inheritance(original_models)
    
    # 检查是否有改动
    if expanded_models != original_models:
        print(f"  ✅ Found models to expand:")
        
        # 显示变化
        for model_id, model_config in expanded_models.items():
            if 'base_model' in model_config:
                original_keys = set(original_models[model_id].keys())
                new_keys = set(model_config.keys())
                added_keys = new_keys - original_keys
                if added_keys:
                    print(f"    🔸 {model_id}: added {list(added_keys)}")
        
        if not dry_run:
            # 创建备份
            backup_path = filepath + '.backup'
            original_data = copy.deepcopy(data)
            if save_json_file(backup_path, original_data):
                print(f"  💾 Backup created: {backup_path}")
            
            # 更新数据并保存
            data['models'] = expanded_models
            if save_json_file(filepath, data):
                print(f"  ✅ Successfully updated {filepath}")
                return True
            else:
                return False
        else:
            print(f"  🔍 [DRY RUN] Would update {filepath}")
            return True
    else:
        print(f"  ✅ No changes needed")
        return True

def find_json_files() -> list:
    """查找所有相关的JSON配置文件"""
    patterns = [
        "**/*.json",
        "**/scheduler_*/*.json",
        "**/test_*.json"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # 去重并过滤
    unique_files = list(set(all_files))
    config_files = []
    
    for filepath in unique_files:
        # 跳过明显不相关的文件
        filename = os.path.basename(filepath).lower()
        skip_patterns = ['backup', 'log', 'output', 'system_stats', 'experiment_results']
        if any(pattern in filename for pattern in skip_patterns):
            continue
        
        # 检查是否包含models字段
        data = load_json_file(filepath)
        if data and 'models' in data:
            config_files.append(filepath)
    
    return sorted(config_files)

def main():
    """主函数"""
    print("🔧 JSON模型配置修正脚本")
    print("📋 功能：为使用base_model的模型添加完整参数")
    print("=" * 60)
    
    # 查找配置文件
    config_files = find_json_files()
    
    if not config_files:
        print("❌ 未找到包含模型配置的JSON文件！")
        return
    
    print(f"📁 找到 {len(config_files)} 个模型配置文件:")
    for f in config_files:
        print(f"  • {f}")
    
    # 询问是否预览
    print(f"\n🔍 先进行预览模式 (查看会做什么修改)...")
    
    # 预览模式
    print("\n" + "=" * 40 + " 预览模式 " + "=" * 40)
    preview_success = 0
    for filepath in config_files:
        if process_config_file(filepath, dry_run=True):
            preview_success += 1
    
    # 询问是否执行
    print(f"\n📊 预览完成: {preview_success}/{len(config_files)} 个文件可以处理")
    
    response = input("\n🚀 是否执行实际修改? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 操作已取消")
        return
    
    # 执行实际修改
    print("\n" + "=" * 40 + " 执行修改 " + "=" * 40)
    success_count = 0
    for filepath in config_files:
        if process_config_file(filepath, dry_run=False):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 成功处理 {success_count}/{len(config_files)} 个文件")
    
    if success_count < len(config_files):
        print("⚠️  部分文件处理失败，请检查上面的错误信息")
    else:
        print("🎉 所有文件处理完成！")

if __name__ == "__main__":
    main()