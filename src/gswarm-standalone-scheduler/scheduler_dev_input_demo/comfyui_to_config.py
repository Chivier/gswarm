#!/usr/bin/env python3
"""
ComfyUI Workflow to Config Converter

This script converts ComfyUI workflow JSON files to the config format
required by the scheduler development input demo.

Usage:
    uv run comfyui_to_config.py --workflow path/to/workflow.json --output config.json
    uv run comfyui_to_config.py --workflow-dir path/to/workflows/ --output-dir configs/
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class ComfyUIToConfigConverter:
    """Converts ComfyUI workflows to scheduler config format."""
    
    # Model type mappings based on ComfyUI node types
    MODEL_TYPE_MAPPING = {
        # Text/Language models
        'CLIPTextEncode': 'embedding',
        'CLIPTextEncodeSDXL': 'embedding', 
        'CLIPTextEncodeFlux': 'embedding',
        'TripleCLIPLoader': 'embedding',
        'DualCLIPLoader': 'embedding',
        'CLIPLoader': 'embedding',
        
        # Image generation models
        'KSampler': 'image_generation',
        'KSamplerAdvanced': 'image_generation',
        'SamplerCustom': 'image_generation',
        'UNETLoader': 'image_generation',
        'CheckpointLoaderSimple': 'image_generation',
        'FluxGuidance': 'image_generation',
        
        # VAE/Encoding models
        'VAEDecode': 'image_processing',
        'VAEEncode': 'image_processing',
        'VAELoader': 'image_processing',
        
        # Control/Conditioning
        'ControlNetLoader': 'control_net',
        'ControlNetApply': 'control_net',
        'ControlNetApplyAdvanced': 'control_net',
        
        # Upscaling
        'UpscaleModelLoader': 'upscaling',
        'ImageUpscaleWithModel': 'upscaling',
        
        # Video
        'VideoLinearCFGGuidance': 'video_generation',
        'HunyuanVideoSampler': 'video_generation',
    }
    
    # Memory requirements by model type (GB)
    MEMORY_REQUIREMENTS = {
        'embedding': 2,
        'image_generation': 8,
        'image_processing': 4,
        'control_net': 6,
        'upscaling': 6,
        'video_generation': 16,
        'llm': 14,
    }
    
    # Inference time estimates by model type (seconds)
    INFERENCE_TIMES = {
        'embedding': {'mean': 0.5, 'std': 0.1},
        'image_generation': {'mean': 3.0, 'std': 0.8},
        'image_processing': {'mean': 1.0, 'std': 0.3},
        'control_net': {'mean': 2.0, 'std': 0.5},
        'upscaling': {'mean': 4.0, 'std': 1.0},
        'video_generation': {'mean': 15.0, 'std': 3.0},
        'llm': {'tokens_per_second': 50, 'token_mean': 512, 'token_std': 128},
    }

    def __init__(self):
        self.models = {}
        self.workflows = {}

    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Load ComfyUI workflow JSON file."""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_model_from_node(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract model definition from a ComfyUI node."""
        node_type = node.get('type', '')
        
        # Skip nodes that don't represent models
        if node_type not in self.MODEL_TYPE_MAPPING:
            return None
            
        model_type = self.MODEL_TYPE_MAPPING[node_type]
        
        # Generate model name based on node type and widgets
        model_name = self._generate_model_name(node)
        
        # Get memory and performance characteristics
        memory_gb = self.MEMORY_REQUIREMENTS.get(model_type, 8)
        inference_config = self.INFERENCE_TIMES.get(model_type, {'mean': 2.0, 'std': 0.5})
        
        model_def = {
            'name': model_name,
            'type': model_type,
            'memory_gb': memory_gb,
            'gpus_required': 1,
        }
        
        # Add type-specific configuration
        if model_type == 'llm':
            model_def.update({
                'tokens_per_second': inference_config['tokens_per_second'],
                'token_mean': inference_config['token_mean'],
                'token_std': inference_config['token_std']
            })
        else:
            model_def.update({
                'inference_time_mean': inference_config['mean'],
                'inference_time_std': inference_config['std']
            })
            
        return model_def

    def _generate_model_name(self, node: Dict[str, Any]) -> str:
        """Generate a descriptive model name from node information."""
        node_type = node.get('type', 'unknown')
        
        # Try to extract model name from widgets_values
        widgets = node.get('widgets_values', [])
        if widgets:
            # First widget often contains model filename
            model_file = str(widgets[0])
            if model_file and model_file != 'None':
                # Clean up filename
                name = Path(model_file).stem
                return name.replace('_', ' ').replace('-', ' ').title()
        
        # Fallback to node type
        return node_type.replace('Loader', '').replace('Net', ' Net')

    def build_workflow_graph(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build workflow graph from ComfyUI nodes and links."""
        nodes_data = workflow_data.get('nodes', [])
        
        # Create workflow nodes
        workflow_nodes = []
        node_id_map = {}  # Map ComfyUI node IDs to our node IDs
        
        for i, node in enumerate(nodes_data):
            node_id = f"node{i+1}"
            node_id_map[node['id']] = node_id
            
            # Extract model for this node
            model_def = self.extract_model_from_node(node)
            if not model_def:
                continue
                
            model_key = self._get_model_key(model_def['name'])
            self.models[model_key] = model_def
            
            # Determine inputs and outputs based on node structure
            inputs = self._extract_node_inputs(node)
            outputs = self._extract_node_outputs(node)
            config_options = self._extract_config_options(node)
            
            workflow_node = {
                'id': node_id,
                'model': model_key,
                'inputs': inputs,
                'outputs': outputs
            }
            
            if config_options:
                workflow_node['config_options'] = config_options
                
            workflow_nodes.append(workflow_node)
        
        # Build edges from node connections
        edges = self._build_edges(nodes_data, node_id_map)
        
        return {
            'nodes': workflow_nodes,
            'edges': edges
        }

    def _get_model_key(self, model_name: str) -> str:
        """Generate a key for the model dictionary."""
        return model_name.lower().replace(' ', '_').replace('-', '_')

    def _extract_node_inputs(self, node: Dict[str, Any]) -> List[str]:
        """Extract input names from ComfyUI node."""
        inputs = []
        for inp in node.get('inputs', []):
            if inp.get('name'):
                inputs.append(inp['name'].lower())
        
        # Add default inputs based on node type
        node_type = node.get('type', '')
        if 'TextEncode' in node_type:
            inputs.append('text_prompt')
        elif node_type == 'EmptyLatentImage':
            inputs.append('image_dimensions')
            
        return inputs or ['input_data']

    def _extract_node_outputs(self, node: Dict[str, Any]) -> List[str]:
        """Extract output names from ComfyUI node."""
        outputs = []
        for out in node.get('outputs', []):
            if out.get('name'):
                outputs.append(out['name'].lower())
                
        return outputs or ['output_data']

    def _extract_config_options(self, node: Dict[str, Any]) -> List[str]:
        """Extract configurable options from node widgets."""
        config_options = []
        
        # Common configurable parameters
        widgets = node.get('widgets_values', [])
        node_type = node.get('type', '')
        
        if node_type == 'EmptyLatentImage' and len(widgets) >= 2:
            config_options.extend(['width', 'height'])
        elif 'Sampler' in node_type:
            config_options.extend(['steps', 'cfg', 'seed'])
        elif 'TextEncode' in node_type:
            config_options.append('prompt')
            
        return config_options

    def _build_edges(self, nodes_data: List[Dict], node_id_map: Dict[int, str]) -> List[Dict[str, str]]:
        """Build workflow edges from ComfyUI node connections."""
        edges = []
        
        # Track connections between nodes
        for node in nodes_data:
            node_id = node_id_map.get(node['id'])
            if not node_id:
                continue
                
            for inp in node.get('inputs', []):
                if 'link' in inp and inp['link'] is not None:
                    # Find the source node for this link
                    source_node_id = self._find_source_node(nodes_data, inp['link'], node_id_map)
                    if source_node_id and source_node_id != node_id:
                        edges.append({
                            'from': source_node_id,
                            'to': node_id
                        })
        
        return edges

    def _find_source_node(self, nodes_data: List[Dict], link_id: int, node_id_map: Dict[int, str]) -> Optional[str]:
        """Find the source node for a given link ID."""
        for node in nodes_data:
            for out in node.get('outputs', []):
                links = out.get('links')
                if links and link_id in links:
                    return node_id_map.get(node['id'])
        return None

    def convert_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Convert a single ComfyUI workflow to config format."""
        workflow_data = self.load_workflow(workflow_path)
        
        # Generate workflow name from filename
        workflow_name = Path(workflow_path).stem.lower().replace(' ', '_').replace('-', '_')
        
        # Build workflow definition
        workflow_graph = self.build_workflow_graph(workflow_data)
        
        workflow_def = {
            'name': Path(workflow_path).stem,
            'nodes': workflow_graph['nodes'],
            'edges': workflow_graph['edges']
        }
        
        self.workflows[workflow_name] = workflow_def
        
        return {
            'models': dict(self.models),
            'workflows': {workflow_name: workflow_def}
        }

    def convert_multiple_workflows(self, workflow_paths: List[str]) -> Dict[str, Any]:
        """Convert multiple ComfyUI workflows to a single config."""
        for workflow_path in workflow_paths:
            self.convert_workflow(workflow_path)
            
        return {
            'models': dict(self.models),
            'workflows': dict(self.workflows)
        }

    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save config to JSON file."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Config saved to: {output_path}")
        print(f"📊 Generated {len(config['models'])} models and {len(config['workflows'])} workflows")


def main():
    parser = argparse.ArgumentParser(description='Convert ComfyUI workflows to scheduler config format')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--workflow', type=str, help='Path to single ComfyUI workflow JSON file')
    group.add_argument('--workflow-dir', type=str, help='Directory containing ComfyUI workflow JSON files')
    
    parser.add_argument('--output', type=str, help='Output config file path (used for both single workflow and multiple workflows)')
    parser.add_argument('--output-dir', type=str, help='Output directory (only used for multiple workflows when --output is not specified)')
    parser.add_argument('--output-prefix', type=str, default='config', 
                       help='Output filename prefix (only used for multiple workflows when --output is not specified, default: config)')
    
    args = parser.parse_args()
    
    converter = ComfyUIToConfigConverter()
    
    if args.workflow:
        # Convert single workflow
        output_path = args.output or f"{args.output_prefix}.json"
        config = converter.convert_workflow(args.workflow)
        converter.save_config(config, output_path)
        
    elif args.workflow_dir:
        # Convert multiple workflows
        workflow_dir = Path(args.workflow_dir)
        workflow_files = list(workflow_dir.glob("*.json"))
        
        if not workflow_files:
            print(f"❌ No JSON files found in {workflow_dir}")
            return
        
        print(f"🔍 Found {len(workflow_files)} workflow files")
        
        config = converter.convert_multiple_workflows([str(f) for f in workflow_files])
        
        # Use --output if provided, otherwise use output-dir and prefix
        if args.output:
            output_path = args.output
        else:
            output_dir = args.output_dir or "."
            output_path = os.path.join(output_dir, f"{args.output_prefix}.json")
        
        converter.save_config(config, output_path)


if __name__ == "__main__":
    main()