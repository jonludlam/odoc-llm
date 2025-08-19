#!/usr/bin/env python3
"""Fix module type naming in module-descriptions JSON files.

This script fixes incorrectly parsed module type names by:
1. Removing 'module-type-' from module names
2. Replacing all remaining hyphens with dots
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any


def fix_module_name(module_name: str) -> str:
    """Fix module type naming issues in a module name.
    
    1. Removes 'module-type-' from the name
    2. Replaces all remaining hyphens with dots
    
    Examples:
    - 'Archi.S-Component-module-type-COMPONENT' -> 'Archi.S.Component.COMPONENT'
    - 'Archi.S-Component' -> 'Archi.S.Component'
    """
    # Remove 'module-type-'
    fixed_name = module_name.replace('module-type-', '')
    
    # Replace all remaining hyphens with dots
    fixed_name = fixed_name.replace('-', '.')
    
    return fixed_name


def fix_module_descriptions_file(file_path: Path) -> Tuple[bool, int]:
    """Fix module names in a single JSON file.
    
    Returns:
        Tuple of (was_modified, number_of_fixes)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    fixes_count = 0
    modified = False
    
    # Navigate through the JSON structure
    if 'libraries' in data:
        for library_name, library_data in data['libraries'].items():
            if 'modules' in library_data and isinstance(library_data['modules'], dict):
                # Create a new modules dict with fixed keys
                new_modules = {}
                for module_name, module_desc in library_data['modules'].items():
                    fixed_name = fix_module_name(module_name)
                    if fixed_name != module_name:
                        fixes_count += 1
                        modified = True
                        print(f"  {module_name} -> {fixed_name}")
                    new_modules[fixed_name] = module_desc
                
                # Replace the modules dict
                library_data['modules'] = new_modules
    
    # Write back if modified
    if modified:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return modified, fixes_count


def main():
    """Process all JSON files in module-descriptions directory."""
    module_desc_dir = Path('module-descriptions')
    
    if not module_desc_dir.exists():
        print(f"Error: {module_desc_dir} directory not found")
        return
    
    total_files = 0
    modified_files = 0
    total_fixes = 0
    
    # Process all JSON files
    for json_file in sorted(module_desc_dir.glob('*.json')):
        total_files += 1
        print(f"\nProcessing {json_file.name}...")
        
        was_modified, fixes = fix_module_descriptions_file(json_file)
        
        if was_modified:
            modified_files += 1
            total_fixes += fixes
            print(f"  Fixed {fixes} module names")
        else:
            print("  No changes needed")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total module names fixed: {total_fixes}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()