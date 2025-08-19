#!/usr/bin/env python3
"""
Fill empty module descriptions in module-descriptions directory.

This script identifies modules with empty descriptions and re-summarizes them
using the existing module description generation infrastructure.
"""

import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the existing module description generator components
from generate_module_descriptions import (
    LLMClient,
    ModuleContent,
    ModuleExtractor
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global shutdown flag
shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Shutdown requested due to signal {signum}")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@dataclass
class EmptyModuleInfo:
    """Information about a module with empty description."""
    package: str
    library: str
    module_path: str
    description: str
    is_module_type: bool
    parent_path: Optional[str] = None

class EmptyDescriptionFiller:
    """Main class for filling empty module descriptions."""
    
    def __init__(self, module_descriptions_dir: Path, parsed_docs_dir: Path, 
                 llm_client: LLMClient, extractor: ModuleExtractor):
        self.module_descriptions_dir = module_descriptions_dir
        self.parsed_docs_dir = parsed_docs_dir
        self.llm_client = llm_client
        self.extractor = extractor
    
    def is_empty_description(self, description: str) -> bool:
        """Check if a description is literally empty."""
        return description.strip() == ""
    
    def find_empty_modules(self, packages: Optional[List[str]] = None) -> Dict[str, List[EmptyModuleInfo]]:
        """Find all modules with empty descriptions, organized by package."""
        empty_modules = defaultdict(list)
        
        logger.info("Scanning for empty module descriptions...")
        
        pattern = "*.json"
        for package_file in self.module_descriptions_dir.glob(pattern):
            package_name = package_file.stem
            
            # Skip if package filter is active and this package isn't included
            if packages and package_name not in packages:
                continue
            
            try:
                with open(package_file, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                
                if 'libraries' not in package_data:
                    continue
                
                for library_name, library_data in package_data['libraries'].items():
                    if 'modules' not in library_data:
                        continue
                    
                    for module_path, module_data in library_data['modules'].items():
                        description = module_data.get('description', '')
                        is_module_type = module_data.get('is_module_type', False)
                        
                        if self.is_empty_description(description):
                            # Determine parent path
                            parent_path = self.get_parent_path(module_path)
                            
                            module_info = EmptyModuleInfo(
                                package=package_name,
                                library=library_name,
                                module_path=module_path,
                                description=description,
                                is_module_type=is_module_type,
                                parent_path=parent_path
                            )
                            empty_modules[package_name].append(module_info)
                            
            except Exception as e:
                logger.warning(f"Failed to process {package_file}: {e}")
        
        total_empty = sum(len(modules) for modules in empty_modules.values())
        logger.info(f"Found {total_empty} empty modules across {len(empty_modules)} packages")
        
        return empty_modules
    
    def get_parent_path(self, module_path: str) -> Optional[str]:
        """Get the parent module path from a module path."""
        parts = module_path.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return None
    
    def load_parsed_modules(self, package: str) -> Dict[str, ModuleContent]:
        """Load and extract all modules from parsed documentation."""
        parsed_file = self.parsed_docs_dir / f"{package}.json"
        
        if not parsed_file.exists():
            logger.warning(f"No parsed docs found for package {package}")
            return {}
        
        try:
            # Extract modules using the existing extractor
            modules_list = self.extractor.extract_from_parsed_json(parsed_file)
            
            # Convert to dictionary keyed by module path
            modules_dict = {}
            for module in modules_list:
                modules_dict[module.path] = module
            
            return modules_dict
            
        except Exception as e:
            logger.warning(f"Failed to load parsed modules for {package}: {e}")
            return {}
    
    def process_package_modules(self, package: str, empty_module_infos: List[EmptyModuleInfo], 
                              max_workers: int = 4, log_prompts: bool = False) -> Dict[str, str]:
        """Process all empty modules in a package and return updates."""
        updates = {}
        
        # Load all parsed modules for the package
        all_modules = self.load_parsed_modules(package)
        if not all_modules:
            logger.warning(f"No parsed modules found for {package}, skipping")
            return updates
        
        # Load existing descriptions for context
        existing_descriptions = self.load_existing_descriptions(package)
        
        # Process modules in dependency order (leaves first)
        sorted_modules = self.sort_modules_by_dependency(empty_module_infos)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for module_info in sorted_modules:
                if shutdown_requested:
                    break
                
                if module_info.module_path not in all_modules:
                    logger.warning(f"Module {module_info.module_path} not found in parsed data")
                    continue
                
                module_content = all_modules[module_info.module_path]
                
                # Submit task
                future = executor.submit(
                    self.generate_single_description,
                    module_content,
                    all_modules,
                    existing_descriptions,
                    log_prompts
                )
                futures.append((future, module_info))
            
            # Collect results
            for future, module_info in futures:
                if shutdown_requested:
                    break
                    
                try:
                    new_description = future.result(timeout=60)
                    if new_description and not self.is_empty_description(new_description):
                        updates[module_info.module_path] = new_description
                        existing_descriptions[module_info.module_path] = new_description
                        logger.info(f"Generated description for {package}/{module_info.module_path}")
                except Exception as e:
                    logger.error(f"Failed to generate description for {package}/{module_info.module_path}: {e}")
        
        return updates
    
    def generate_single_description(self, module: ModuleContent, all_modules: Dict[str, ModuleContent],
                                  existing_descriptions: Dict[str, str], log_prompts: bool) -> str:
        """Generate description for a single module."""
        try:
            # Use the LLM client to generate description
            description = self.llm_client.generate_module_description(
                module,
                all_modules,
                log_prompts,
                existing_descriptions
            )
            return description
        except Exception as e:
            logger.error(f"Error generating description for {module.path}: {e}")
            return ""
    
    def sort_modules_by_dependency(self, modules: List[EmptyModuleInfo]) -> List[EmptyModuleInfo]:
        """Sort modules so that child modules are processed before parents."""
        # Create dependency graph
        children_map = defaultdict(list)
        for module in modules:
            if module.parent_path:
                children_map[module.parent_path].append(module.module_path)
        
        # Topological sort (process leaves first)
        sorted_modules = []
        visited = set()
        
        def visit(module_info):
            if module_info.module_path in visited:
                return
            visited.add(module_info.module_path)
            
            # Visit children first
            for child_path in children_map[module_info.module_path]:
                # Find the child module info
                for m in modules:
                    if m.module_path == child_path:
                        visit(m)
                        break
            
            sorted_modules.append(module_info)
        
        for module in modules:
            visit(module)
        
        return sorted_modules
    
    def load_existing_descriptions(self, package: str) -> Dict[str, str]:
        """Load existing module descriptions for a package."""
        descriptions = {}
        package_file = self.module_descriptions_dir / f"{package}.json"
        
        if not package_file.exists():
            return descriptions
        
        try:
            with open(package_file, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            for library_data in package_data.get('libraries', {}).values():
                for module_path, module_data in library_data.get('modules', {}).items():
                    description = module_data.get('description', '')
                    if description and not self.is_empty_description(description):
                        descriptions[module_path] = description
        
        except Exception as e:
            logger.warning(f"Failed to load existing descriptions for {package}: {e}")
        
        return descriptions
    
    def update_parents_for_package(self, package: str, child_updates: Dict[str, str],
                                  max_workers: int = 4, log_prompts: bool = False) -> Dict[str, str]:
        """Update parent module descriptions after children have been updated."""
        parent_updates = {}
        
        # Find unique parent paths
        parent_paths = set()
        for module_path in child_updates.keys():
            parent_path = self.get_parent_path(module_path)
            if parent_path:
                parent_paths.add(parent_path)
        
        if not parent_paths:
            return parent_updates
        
        # Load all modules and existing descriptions
        all_modules = self.load_parsed_modules(package)
        existing_descriptions = self.load_existing_descriptions(package)
        # Merge in the child updates
        existing_descriptions.update(child_updates)
        
        logger.info(f"Updating {len(parent_paths)} parent modules for {package}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for parent_path in parent_paths:
                if shutdown_requested:
                    break
                
                if parent_path not in all_modules:
                    logger.warning(f"Parent module {parent_path} not found in parsed data")
                    continue
                
                parent_module = all_modules[parent_path]
                
                # Submit task to regenerate parent with updated children
                future = executor.submit(
                    self.generate_single_description,
                    parent_module,
                    all_modules,
                    existing_descriptions,
                    log_prompts
                )
                futures.append((future, parent_path))
            
            # Collect results
            for future, parent_path in futures:
                if shutdown_requested:
                    break
                    
                try:
                    new_description = future.result(timeout=60)
                    if new_description and not self.is_empty_description(new_description):
                        parent_updates[parent_path] = new_description
                        logger.info(f"Updated parent description for {package}/{parent_path}")
                except Exception as e:
                    logger.error(f"Failed to update parent {package}/{parent_path}: {e}")
        
        return parent_updates
    
    def apply_updates_to_file(self, package: str, updates: Dict[str, str]) -> None:
        """Apply description updates to a package's module description file."""
        if not updates:
            return
            
        package_file = self.module_descriptions_dir / f"{package}.json"
        
        if not package_file.exists():
            logger.warning(f"Package file {package_file} not found, skipping updates")
            return
        
        try:
            # Load current data
            with open(package_file, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            # Apply updates
            updated_count = 0
            for library_data in package_data.get('libraries', {}).values():
                for module_path, module_data in library_data.get('modules', {}).items():
                    if module_path in updates:
                        module_data['description'] = updates[module_path]
                        updated_count += 1
            
            # Save updated data
            with open(package_file, 'w', encoding='utf-8') as f:
                json.dump(package_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Applied {updated_count} updates to {package}")
            
        except Exception as e:
            logger.error(f"Failed to apply updates to {package}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fill empty module descriptions')
    parser.add_argument('--module-descriptions-dir', type=Path, default=Path('module-descriptions'),
                       help='Directory containing module description JSON files')
    parser.add_argument('--parsed-docs-dir', type=Path, default=Path('parsed-docs'),
                       help='Directory containing parsed documentation JSON files')
    parser.add_argument('--llm-url', default='https://openrouter.ai/api',
                       help='Base URL for LLM server (without /v1)')
    parser.add_argument('--model', default='qwen/qwen3-235b-a22b',
                       help='LLM model name')
    parser.add_argument('--api-key', 
                       help='API key for LLM service (defaults to TOKEN file, then OPENROUTER_API_KEY env var)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--packages', 
                       help='Comma-separated list of packages to process (default: all)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of packages to process (for testing)')
    parser.add_argument('--log-prompts', action='store_true',
                       help='Log LLM prompts and responses')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    if not args.module_descriptions_dir.exists():
        logger.error(f"Module descriptions directory not found: {args.module_descriptions_dir}")
        return 1
    
    if not args.parsed_docs_dir.exists():
        logger.error(f"Parsed docs directory not found: {args.parsed_docs_dir}")
        return 1
    
    try:
        # Handle API key
        import os
        api_key = args.api_key
        if api_key is None:
            # First try to read from TOKEN file
            token_file = Path("TOKEN")
            if token_file.exists():
                try:
                    api_key = token_file.read_text().strip()
                    logger.info("API key loaded from TOKEN file")
                except Exception as e:
                    logger.warning(f"Failed to read TOKEN file: {e}")
            
            # Fall back to environment variable
            if not api_key:
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if api_key:
                    logger.info("API key loaded from OPENROUTER_API_KEY environment variable")
            
            if not api_key:
                api_key = "dummy_key"  # Fall back to dummy for local servers
        
        # Initialize components
        llm_client = LLMClient(args.llm_url, args.model, api_key)
        extractor = ModuleExtractor()
        
        # Initialize filler
        filler = EmptyDescriptionFiller(
            args.module_descriptions_dir,
            args.parsed_docs_dir,
            llm_client,
            extractor
        )
        
        # Find empty modules
        packages_filter = None
        if args.packages:
            packages_filter = args.packages.split(',')
        
        empty_modules = filler.find_empty_modules(packages_filter)
        
        # Apply limit if specified
        if args.limit:
            limited_modules = {}
            count = 0
            for package, modules in empty_modules.items():
                if count >= args.limit:
                    break
                limited_modules[package] = modules
                count += 1
            empty_modules = limited_modules
        
        total_empty = sum(len(modules) for modules in empty_modules.values())
        if total_empty == 0:
            logger.info("No empty modules found")
            return 0
        
        logger.info(f"Processing {total_empty} empty modules across {len(empty_modules)} packages")
        
        if args.dry_run:
            logger.info("DRY RUN: Would update the following modules:")
            for package, modules in empty_modules.items():
                logger.info(f"Package {package}:")
                for module in modules:
                    logger.info(f"  - {module.module_path}")
            return 0
        
        # Process each package
        with tqdm(total=len(empty_modules), desc="Processing packages") as pbar:
            for package, module_infos in empty_modules.items():
                if shutdown_requested:
                    break
                
                logger.info(f"Processing {len(module_infos)} empty modules in {package}")
                
                # Generate descriptions for empty modules
                child_updates = filler.process_package_modules(
                    package, module_infos, args.workers, args.log_prompts
                )
                
                if shutdown_requested:
                    break
                
                # Update parent modules if there were child updates
                if child_updates:
                    parent_updates = filler.update_parents_for_package(
                        package, child_updates, args.workers, args.log_prompts
                    )
                    
                    # Merge parent updates into child updates
                    all_updates = {**child_updates, **parent_updates}
                    
                    # Apply all updates to the file
                    filler.apply_updates_to_file(package, all_updates)
                    
                    logger.info(f"Completed {package}: {len(all_updates)} total updates")
                else:
                    logger.info(f"No updates generated for {package}")
                
                pbar.update(1)
        
        if shutdown_requested:
            logger.info("Shutdown requested, stopping...")
            return 1
        
        logger.info("Successfully completed filling empty descriptions")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fill empty descriptions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())