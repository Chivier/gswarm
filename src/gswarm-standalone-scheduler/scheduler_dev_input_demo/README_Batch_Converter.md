# ComfyUI Batch Converter - One-to-One Workflow Conversion

这个脚本提供了一种将ComfyUI workflow一对一转换为单独config文件的方法，而不是将所有workflow合并到一个config中。

## 功能特点

- **一对一转换**: 每个ComfyUI workflow JSON文件生成一个独立的config文件
- **批量处理**: 可以处理整个目录中的所有workflow文件
- **自定义命名**: 支持自定义文件名前缀和后缀
- **错误处理**: 继续处理其他文件，即使某个文件转换失败
- **详细报告**: 提供转换过程的详细统计和摘要

## 与原始转换器的区别

| 特性 | `comfyui_to_config.py` | `comfyui_batch_converter.py` |
|------|------------------------|------------------------------|
| 转换方式 | 多个workflow → 单个config | 每个workflow → 独立config |
| 输出文件 | 1个合并的config文件 | N个独立的config文件 |
| 适用场景 | 统一调度多个workflow | 独立测试/部署每个workflow |
| 模型共享 | 所有workflow共享模型定义 | 每个config有独立模型定义 |

## 使用方法

### 基本用法

```bash
# 转换单个workflow
uv run comfyui_batch_converter.py --workflow path/to/workflow.json --output-dir configs/

# 转换目录中的所有workflow
uv run comfyui_batch_converter.py --workflow-dir path/to/workflows/ --output-dir configs/
```

### 高级选项

```bash
# 使用自定义命名模式
uv run comfyui_batch_converter.py \
  --workflow-dir workflows/ \
  --output-dir configs/ \
  --prefix "model_" \
  --suffix "_config"

# 指定特定的输出文件名（仅适用于单个workflow）
uv run comfyui_batch_converter.py \
  --workflow workflow.json \
  --output-dir configs/ \
  --output-file my_custom_config.json
```

### 参数说明

- `--workflow`: 单个ComfyUI workflow JSON文件路径
- `--workflow-dir`: 包含多个workflow文件的目录
- `--output-dir`: 输出目录（默认: `configs`）
- `--output-file`: 特定输出文件名（仅用于单个workflow）
- `--prefix`: 生成文件名的前缀
- `--suffix`: 生成文件名的后缀（在.json之前）
- `--continue-on-error`: 遇到错误时继续处理其他文件

## 输出文件命名

生成的config文件名格式为：
```
{prefix}{sanitized_workflow_name}{suffix}.json
```

例如：
- 原文件: `FLUX.1 SCHNELL 1.0.json`
- 生成文件: `comfyui_FLUX.1_SCHNELL_1.0_scheduler.json`
  - prefix: `comfyui_`
  - sanitized name: `FLUX.1_SCHNELL_1.0`
  - suffix: `_scheduler`

## 文件名清理规则

为了确保生成的文件名兼容各种文件系统，脚本会自动清理workflow名称：
- 移除特殊字符（保留字母、数字、空格、连字符、下划线）
- 将空格替换为下划线
- 移除首尾的下划线

## 示例输出

```bash
$ uv run comfyui_batch_converter.py --workflow-dir workflows/ --output-dir configs/

🔄 ComfyUI Batch Converter - One-to-One Workflow Conversion
📁 Output directory: configs
📁 Converting workflows from directory: workflows/
🔍 Found 3 workflow files in workflows/
📁 Output directory: configs

📄 [1/3] Converting: FLUX.1 SCHNELL 1.0.json
✅ Config saved to: configs/FLUX.1_SCHNELL_1.0.json
📊 Generated 5 models and 1 workflows

📄 [2/3] Converting: SD3 BASE 1.0.json
✅ Config saved to: configs/SD3_BASE_1.0.json
📊 Generated 6 models and 1 workflows

📄 [3/3] Converting: Stable Cascade Canny ControlNet.json
✅ Config saved to: configs/Stable_Cascade_Canny_ControlNet.json
📊 Generated 9 models and 1 workflows

============================================================
📊 Batch Conversion Summary
============================================================
📁 Output Directory: configs
📄 Files Processed: 3
✅ Successful: 3
❌ Failed: 0
🎯 Total Models: 20
📋 Config Files Created: 3
```

## 使用生成的Config文件

每个生成的config文件都可以独立使用：

```bash
# 使用单个config生成请求序列
uv run model_seq_gen.py --config configs/FLUX.1_SCHNELL_1.0.json --num-requests 10

# 测试config文件的有效性
uv run test_individual_configs.py
```

## 使用场景

### 1. 独立模型测试
当你需要测试每个workflow的性能时：
```bash
uv run comfyui_batch_converter.py --workflow-dir workflows/ --output-dir test_configs/
# 然后分别测试每个config
for config in test_configs/*.json; do
    uv run model_seq_gen.py --config "$config" --num-requests 5
done
```

### 2. 渐进式部署
逐步部署不同的workflow到生产环境：
```bash
# 只部署FLUX相关的workflow
uv run model_seq_gen.py --config configs/comfyui_FLUX*_scheduler.json --num-requests 100
```

### 3. A/B测试
比较不同workflow版本的性能：
```bash
# 测试两个版本的FLUX workflow
uv run model_seq_gen.py --config configs/FLUX_v1.json --output-prefix test_v1
uv run model_seq_gen.py --config configs/FLUX_v2.json --output-prefix test_v2
```

## 注意事项

1. **模型重复**: 每个config文件包含独立的模型定义，可能存在重复
2. **文件大小**: 相比合并版本，会生成更多较小的文件
3. **内存使用**: 每个workflow的模型不会与其他workflow共享
4. **维护成本**: 需要管理多个config文件而不是一个

## 与测试工具集成

使用提供的测试脚本验证生成的config文件：

```bash
# 测试所有生成的individual config文件
uv run test_individual_configs.py

# 只测试前3个config文件
uv run python -c "
from test_individual_configs import test_individual_configs
test_individual_configs(max_configs=3)
"
```

## 故障排除

### 常见问题

1. **没有找到workflow文件**
   ```
   ❌ No JSON files found in /path/to/workflows
   ```
   确保目录中包含`.json`文件

2. **某些workflow转换失败**
   ```
   ❌ Failed to convert workflow.json: Invalid JSON format
   ```
   检查workflow文件是否为有效的JSON格式

3. **输出目录权限问题**
   ```
   Permission denied: /output/directory
   ```
   确保对输出目录有写权限

### 调试技巧

1. **使用单个文件测试**:
   ```bash
   uv run comfyui_batch_converter.py --workflow problematic_workflow.json --output-dir debug/
   ```

2. **检查原始workflow结构**:
   ```bash
   cat workflow.json | jq '.' # 验证JSON格式
   cat workflow.json | jq '.nodes | length' # 检查节点数量
   ```

3. **详细错误信息**:
   脚本会显示每个失败workflow的具体错误信息，用于调试转换问题。
