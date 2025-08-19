#!/usr/bin/env python3
"""
One-off script to backfill is_module_type information to existing module descriptions.
This reads from parsed-docs/ and updates module-descriptions/ files to include
the is_module_type field for each module.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse
import sys
from tqdm import tqdm

def load_parsed_doc(parsed_file: Path) -> Dict[str, Any]:
    """Load parsed documentation JSON."""
    with open(parsed_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_module_type_info(parsed_data: Dict[str, Any]) -> Dict[str, bool]:
    """Extract module path -> is_module_type mapping from parsed data."""
    module_type_info = {}
    
    # Process modules list
    for module in parsed_data.get('modules', []):
        module_path = module.get('module_path', [])
        if isinstance(module_path, list):
            module_path = '.'.join(module_path)
        
        is_module_type = module.get('is_module_type', False)
        module_type_info[module_path] = is_module_type
    
    return module_type_info

def update_module_descriptions(desc_file: Path, module_type_info: Dict[str, bool], package_name: str) -> bool:
    """Update module descriptions file with is_module_type information."""
    try:
        # Load existing descriptions
        with open(desc_file, 'r', encoding='utf-8') as f:
            desc_data = json.load(f)
        
        updated = False
        
        # Update each library
        for library_name, library_data in desc_data.get('libraries', {}).items():
            modules = library_data.get('modules', {})
            
            # Convert old format to new format if needed
            for module_path, module_info in modules.items():
                if isinstance(module_info, str):
                    # Old format - just a string description
                    # Look up is_module_type from parsed data
                    is_module_type = module_type_info.get(module_path, False)
                    
                    # Handle package prefix variations
                    if not module_path in module_type_info:
                        # Try without package prefix
                        if module_path.startswith(f"{package_name}."):
                            alt_path = module_path[len(package_name)+1:]
                            is_module_type = module_type_info.get(alt_path, False)
                        # Try with unnamed prefix
                        elif not module_path.startswith("unnamed."):
                            alt_path = f"unnamed.{module_path}"
                            is_module_type = module_type_info.get(alt_path, False)
                    
                    modules[module_path] = {
                        "description": module_info,
                        "is_module_type": is_module_type
                    }
                    updated = True
                elif isinstance(module_info, dict) and 'is_module_type' not in module_info:
                    # New format but missing is_module_type
                    is_module_type = module_type_info.get(module_path, False)
                    
                    # Handle package prefix variations
                    if not module_path in module_type_info:
                        if module_path.startswith(f"{package_name}."):
                            alt_path = module_path[len(package_name)+1:]
                            is_module_type = module_type_info.get(alt_path, False)
                        elif not module_path.startswith("unnamed."):
                            alt_path = f"unnamed.{module_path}"
                            is_module_type = module_type_info.get(alt_path, False)
                    
                    module_info['is_module_type'] = is_module_type
                    updated = True
        
        # Write back if updated
        if updated:
            # Write to temp file first for safety
            temp_file = desc_file.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(desc_data, f, indent=2, ensure_ascii=False)
            
            # Atomically replace
            temp_file.rename(desc_file)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {desc_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Backfill is_module_type information to module descriptions')
    parser.add_argument('--package', help='Process only a specific package')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    args = parser.parse_args()
    
    parsed_dir = Path('parsed-docs')
    desc_dir = Path('module-descriptions')
    
    if not parsed_dir.exists():
        print(f"Error: {parsed_dir} directory not found")
        return 1
    
    if not desc_dir.exists():
        print(f"Error: {desc_dir} directory not found")
        return 1
    
    # Get list of packages to process
    if args.package:
        desc_files = [desc_dir / f"{args.package}.json"]
        if not desc_files[0].exists():
            print(f"Error: No description file found for package {args.package}")
            return 1
    else:
        desc_files = list(desc_dir.glob("*.json"))
    
    print(f"Processing {len(desc_files)} package description files...")
    
    updated_count = 0
    error_count = 0
    
    for desc_file in tqdm(desc_files):
        package_name = desc_file.stem
        parsed_file = parsed_dir / f"{package_name}.json"
        
        if not parsed_file.exists():
            print(f"Warning: No parsed file found for {package_name}, skipping")
            continue
        
        try:
            # Load parsed data and extract module type info
            parsed_data = load_parsed_doc(parsed_file)
            module_type_info = extract_module_type_info(parsed_data)
            
            if args.dry_run:
                # Just check if it would be updated
                with open(desc_file, 'r', encoding='utf-8') as f:
                    desc_data = json.load(f)
                
                needs_update = False
                for library_data in desc_data.get('libraries', {}).values():
                    for module_info in library_data.get('modules', {}).values():
                        if isinstance(module_info, str) or (isinstance(module_info, dict) and 'is_module_type' not in module_info):
                            needs_update = True
                            break
                
                if needs_update:
                    print(f"Would update: {package_name}")
                    # Show some examples of what would be updated
                    example_count = 0
                    for lib_name, library_data in desc_data.get('libraries', {}).items():
                        for mod_path, module_info in library_data.get('modules', {}).items():
                            if isinstance(module_info, str):
                                is_mt = module_type_info.get(mod_path, False)
                                # Try alternate paths if not found
                                if not mod_path in module_type_info:
                                    if mod_path.startswith(f"{package_name}."):
                                        alt_path = mod_path[len(package_name)+1:]
                                        is_mt = module_type_info.get(alt_path, False)
                                    elif not mod_path.startswith("unnamed."):
                                        alt_path = f"unnamed.{mod_path}"
                                        is_mt = module_type_info.get(alt_path, False)
                                
                                if is_mt and example_count < 5:  # Show first 5 module types
                                    print(f"  Module type found: {mod_path}")
                                    example_count += 1
                    updated_count += 1
            else:
                # Actually update the file
                if update_module_descriptions(desc_file, module_type_info, package_name):
                    updated_count += 1
                    
        except Exception as e:
            print(f"Error processing {package_name}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(desc_files)}")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())