#!/usr/bin/env python3
"""
ComfyUI Batch Converter - One-to-One Workflow to Config Conversion

This script converts each ComfyUI workflow JSON file to its own individual config file,
creating a one-to-one mapping instead of merging all workflows into a single config.

Usage:
    # Convert single workflow to individual config
    uv run comfyui_batch_converter.py --workflow path/to/workflow.json --output-dir configs/
    
    # Convert all workflows in directory to individual configs
    uv run comfyui_batch_converter.py --workflow-dir path/to/workflows/ --output-dir configs/
    
    # Convert with custom naming pattern
    uv run comfyui_batch_converter.py --workflow-dir workflows/ --output-dir configs/ --prefix "config_" --suffix "_v1"
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Import the converter class from the main script
from comfyui_to_config import ComfyUIToConfigConverter


class ComfyUIBatchConverter:
    """Converts ComfyUI workflows to individual config files (one-to-one mapping)."""
    
    def __init__(self, output_dir: str = "configs", prefix: str = "", suffix: str = ""):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.suffix = suffix
        self.converted_files = []
        self.failed_files = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use as config filename."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[^\w\s\-_.]', '', filename)
        sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
        return sanitized
    
    def generate_output_filename(self, workflow_path: str) -> str:
        """Generate output config filename based on workflow filename."""
        workflow_name = Path(workflow_path).stem
        sanitized_name = self.sanitize_filename(workflow_name)
        
        # Construct filename with prefix and suffix
        filename = f"{self.prefix}{sanitized_name}{self.suffix}.json"
        return str(self.output_dir / filename)
    
    def convert_single_workflow(self, workflow_path: str, output_path: Optional[str] = None) -> bool:
        """Convert a single workflow to its own config file."""
        try:
            # Create a fresh converter instance for each workflow
            converter = ComfyUIToConfigConverter()
            
            # Convert the workflow
            config = converter.convert_workflow(workflow_path)
            
            # Determine output path
            if output_path is None:
                output_path = self.generate_output_filename(workflow_path)
            
            # Save the config
            converter.save_config(config, output_path)
            
            # Track success
            self.converted_files.append({
                'workflow': workflow_path,
                'config': output_path,
                'models': len(config['models']),
                'workflows': len(config['workflows'])
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to convert {workflow_path}: {str(e)}")
            self.failed_files.append({
                'workflow': workflow_path,
                'error': str(e)
            })
            return False
    
    def convert_workflow_directory(self, workflow_dir: str) -> Dict[str, Any]:
        """Convert all workflows in a directory to individual config files."""
        workflow_path = Path(workflow_dir)
        
        if not workflow_path.exists():
            raise ValueError(f"Workflow directory does not exist: {workflow_dir}")
        
        # Find all JSON files
        workflow_files = list(workflow_path.glob("*.json"))
        
        if not workflow_files:
            print(f"⚠️  No JSON files found in {workflow_dir}")
            return self.get_summary()
        
        print(f"🔍 Found {len(workflow_files)} workflow files in {workflow_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Convert each workflow individually
        for i, workflow_file in enumerate(workflow_files, 1):
            print(f"\n📄 [{i}/{len(workflow_files)}] Converting: {workflow_file.name}")
            self.convert_single_workflow(str(workflow_file))
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversion summary statistics."""
        total_models = sum(f['models'] for f in self.converted_files)
        total_workflows = len(self.converted_files)
        
        return {
            'total_files_processed': len(self.converted_files) + len(self.failed_files),
            'successful_conversions': len(self.converted_files),
            'failed_conversions': len(self.failed_files),
            'total_models_generated': total_models,
            'total_config_files': total_workflows,
            'converted_files': self.converted_files,
            'failed_files': self.failed_files,
            'output_directory': str(self.output_dir)
        }
    
    def print_summary(self):
        """Print a detailed summary of the conversion process."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("📊 Batch Conversion Summary")
        print(f"{'='*60}")
        
        print(f"📁 Output Directory: {summary['output_directory']}")
        print(f"📄 Files Processed: {summary['total_files_processed']}")
        print(f"✅ Successful: {summary['successful_conversions']}")
        print(f"❌ Failed: {summary['failed_conversions']}")
        print(f"🎯 Total Models: {summary['total_models_generated']}")
        print(f"📋 Config Files Created: {summary['total_config_files']}")
        
        if self.converted_files:
            print(f"\n✅ Successfully Converted Files:")
            for file_info in self.converted_files:
                workflow_name = Path(file_info['workflow']).name
                config_name = Path(file_info['config']).name
                print(f"  • {workflow_name} → {config_name} ({file_info['models']} models)")
        
        if self.failed_files:
            print(f"\n❌ Failed Conversions:")
            for file_info in self.failed_files:
                workflow_name = Path(file_info['workflow']).name
                print(f"  • {workflow_name}: {file_info['error']}")
        
        print(f"\n📚 Next Steps:")
        print(f"• Use individual config files with: uv run model_seq_gen.py --config <config_file>")
        print(f"• Test configs with: uv run test_converter.py")
        print(f"• Find configs in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ComfyUI workflows to individual config files (one-to-one mapping)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single workflow
  uv run comfyui_batch_converter.py --workflow workflow.json --output-dir configs/
  
  # Convert all workflows in directory
  uv run comfyui_batch_converter.py --workflow-dir workflows/ --output-dir configs/
  
  # Convert with custom naming
  uv run comfyui_batch_converter.py --workflow-dir workflows/ --output-dir configs/ --prefix "model_" --suffix "_config"
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--workflow', type=str, 
                      help='Path to single ComfyUI workflow JSON file')
    group.add_argument('--workflow-dir', type=str, 
                      help='Directory containing ComfyUI workflow JSON files')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='configs',
                       help='Output directory for generated config files (default: configs)')
    parser.add_argument('--output-file', type=str,
                       help='Specific output filename (only for single workflow conversion)')
    
    # Naming options
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for generated config filenames')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix for generated config filenames (before .json)')
    
    # Processing options
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing other files if one fails')
    
    args = parser.parse_args()
    
    # Create batch converter
    batch_converter = ComfyUIBatchConverter(
        output_dir=args.output_dir,
        prefix=args.prefix,
        suffix=args.suffix
    )
    
    print("🔄 ComfyUI Batch Converter - One-to-One Workflow Conversion")
    print(f"📁 Output directory: {args.output_dir}")
    
    try:
        if args.workflow:
            # Convert single workflow
            print(f"📄 Converting single workflow: {args.workflow}")
            
            output_path = None
            if args.output_file:
                output_path = str(Path(args.output_dir) / args.output_file)
            
            success = batch_converter.convert_single_workflow(args.workflow, output_path)
            if success:
                print("✅ Conversion completed successfully!")
            else:
                print("❌ Conversion failed!")
                return 1
                
        elif args.workflow_dir:
            # Convert directory of workflows
            print(f"📁 Converting workflows from directory: {args.workflow_dir}")
            batch_converter.convert_workflow_directory(args.workflow_dir)
        
        # Print summary
        batch_converter.print_summary()
        
        # Return appropriate exit code
        summary = batch_converter.get_summary()
        if summary['failed_conversions'] > 0:
            if args.continue_on_error or summary['successful_conversions'] > 0:
                print(f"\n⚠️  Completed with {summary['failed_conversions']} errors")
                return 0 if summary['successful_conversions'] > 0 else 1
            else:
                return 1
        
        return 0
        
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
