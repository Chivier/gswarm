# ComfyUI Workflow 转换工具

这个工具可以将 ComfyUI workflow JSON 文件转换为调度器开发输入演示所需的配置格式。

## 功能特性

- ✅ 支持单个 workflow 文件转换
- ✅ 支持批量转换整个目录的 workflow 文件 
- ✅ 自动识别 ComfyUI 节点类型并映射为相应的模型类型
- ✅ 自动生成模型资源需求（GPU 内存、推理时间等）
- ✅ 自动构建 workflow 图结构（节点和边）
- ✅ 提取可配置选项（如提示词、采样步数等）

## 安装和运行

本工具使用 uv 管理依赖，确保在项目根目录下运行：

```bash
# 单个文件转换
uv run comfyui_to_config.py --workflow path/to/workflow.json --output config.json

# 批量转换目录中的所有workflow
uv run comfyui_to_config.py --workflow-dir path/to/workflows/ --output batch_config.json

# 查看帮助
uv run comfyui_to_config.py --help
```

## 使用示例

### 1. 转换单个 workflow

```bash
cd /Users/ray/Projects/gswarm/src/gswarm-standalone-scheduler/scheduler_dev_input_demo

# 转换SDXS workflow
uv run comfyui_to_config.py \
  --workflow /Users/ray/Projects/gswarm/examples/ComfyUI-Workflows-ZHO/workflows/zho/SDXS-512-0.9.json \
  --output sdxs_config.json
```

### 2. 批量转换所有 workflow

```bash
# 转换整个目录
uv run comfyui_to_config.py \
  --workflow-dir /Users/ray/Projects/gswarm/examples/ComfyUI-Workflows-ZHO/workflows/zho \
  --output all_workflows_config.json
```

## 支持的模型类型

工具会自动将 ComfyUI 节点映射为以下模型类型：

| ComfyUI 节点类型 | 模型类型 | 内存需求 | 说明 |
|---|---|---|---|
| CLIPTextEncode | embedding | 2GB | 文本编码模型 |
| UNETLoader | image_generation | 8GB | 图像生成模型 |
| VAEDecode/VAEEncode | image_processing | 4GB | 图像处理模型 |
| ControlNetLoader | control_net | 6GB | 控制网络模型 |
| KSampler | image_generation | 8GB | 采样器模型 |
| VideoLinearCFGGuidance | video_generation | 16GB | 视频生成模型 |

## 输出格式

生成的配置文件包含两个主要部分：

### 模型定义 (models)
```json
{
  "models": {
    "stable_diffusion": {
      "name": "Stable Diffusion",
      "type": "image_generation", 
      "memory_gb": 8,
      "gpus_required": 1,
      "inference_time_mean": 3.0,
      "inference_time_std": 0.8
    }
  }
}
```

### Workflow 定义 (workflows)
```json
{
  "workflows": {
    "my_workflow": {
      "name": "My Workflow",
      "nodes": [
        {
          "id": "node1",
          "model": "stable_diffusion",
          "inputs": ["text_prompt"],
          "outputs": ["generated_image"],
          "config_options": ["steps", "cfg", "seed"]
        }
      ],
      "edges": [
        {
          "from": "node1", 
          "to": "node2"
        }
      ]
    }
  }
}
```

## 转换示例结果

转换后的配置文件可以直接用于调度器开发：

```bash
# 使用转换后的配置生成请求序列
uv run model_seq_gen.py --config sdxs_config.json --num-requests 100
```

这将生成：
- `generated_requests.yaml` - 请求序列文件
- `generated_stats.json` - 统计信息

## 注意事项

1. **模型命名**: 工具会尝试从 ComfyUI 节点的 widgets_values 中提取模型文件名，如果失败则使用节点类型作为模型名
2. **资源估算**: 内存和推理时间是基于模型类型的估算值，实际使用时可能需要调整
3. **连接映射**: 工具会尽力构建节点间的连接关系，但复杂的workflow可能需要手动调整
4. **配置选项**: 自动提取常见的可配置选项，如采样步数、CFG值等

## 故障排除

如果转换失败，请检查：
1. workflow JSON 文件格式是否正确
2. 文件路径是否存在
3. 是否有读写权限

如需自定义模型类型映射或资源配置，可以修改脚本中的 `MODEL_TYPE_MAPPING`、`MEMORY_REQUIREMENTS` 和 `INFERENCE_TIMES` 常量。
