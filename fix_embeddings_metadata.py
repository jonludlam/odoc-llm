#!/usr/bin/env python3
"""Fix module type naming in package-embeddings metadata JSON files.

This script fixes incorrectly parsed module type names in the embeddings metadata by:
1. Removing 'module-type-' from module names
2. Replacing all remaining hyphens with dots

The embeddings themselves are not modified, only the metadata.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any, List


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


def fix_embeddings_metadata_file(metadata_path: Path) -> Tuple[bool, int]:
    """Fix module names in a single embeddings metadata JSON file.
    
    Returns:
        Tuple of (was_modified, number_of_fixes)
    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    fixes_count = 0
    modified = False
    
    # Fix module paths in the modules array
    if 'modules' in data and isinstance(data['modules'], list):
        for module_info in data['modules']:
            if 'module_path' in module_info:
                original_path = module_info['module_path']
                fixed_path = fix_module_name(original_path)
                if fixed_path != original_path:
                    module_info['module_path'] = fixed_path
                    fixes_count += 1
                    modified = True
                    print(f"  {original_path} -> {fixed_path}")
    
    # Write back if modified
    if modified:
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return modified, fixes_count


def main():
    """Process all metadata.json files in package-embeddings/packages directory."""
    embeddings_dir = Path('package-embeddings/packages')
    
    if not embeddings_dir.exists():
        print(f"Error: {embeddings_dir} directory not found")
        return
    
    total_packages = 0
    modified_packages = 0
    total_fixes = 0
    
    # Process all package directories
    for package_dir in sorted(embeddings_dir.iterdir()):
        if package_dir.is_dir():
            metadata_path = package_dir / 'metadata.json'
            if metadata_path.exists():
                total_packages += 1
                print(f"\nProcessing {package_dir.name}/metadata.json...")
                
                was_modified, fixes = fix_embeddings_metadata_file(metadata_path)
                
                if was_modified:
                    modified_packages += 1
                    total_fixes += fixes
                    print(f"  Fixed {fixes} module names")
                else:
                    print("  No changes needed")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total packages processed: {total_packages}")
    print(f"  Packages modified: {modified_packages}")
    print(f"  Total module names fixed: {total_fixes}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()