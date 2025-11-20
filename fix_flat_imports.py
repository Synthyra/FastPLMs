#!/usr/bin/env python3
"""
Fix internal imports in boltzgen_flat files.

This script updates all imports within the boltzgen_flat directory so they
reference the flat structure instead of the original boltzgen module.
"""

import os
import re
from pathlib import Path
import json

# Configuration
BOLTZGEN_FLAT_DIR = Path("boltzgen_automodel/boltzgen_flat")
MAPPING_FILE = Path("migration_mapping.json")

def load_import_mapping():
    """Load the import mapping from the migration."""
    if not MAPPING_FILE.exists():
        print(f"ERROR: Mapping file not found: {MAPPING_FILE}")
        print("Please run migrate_boltzgen.py first!")
        return None
    
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
    
    return data['import_mapping']

def fix_imports_in_flat_file(file_path, import_mapping):
    """
    Fix imports in a single file within boltzgen_flat.
    
    Handles patterns like:
    - from boltzgen.X.Y.Z import ...
    - from boltzgen.src.boltzgen.X.Y.Z import ...
    - import boltzgen.X.Y.Z
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Pattern 1: from boltzgen.src.boltzgen.X.Y.Z import ...
    # Pattern 2: from boltzgen.X.Y.Z import ...
    for old_module, new_module in import_mapping.items():
        patterns = [
            (f"from boltzgen.src.boltzgen.{old_module}", f"from boltzgen_flat.{new_module}"),
            (f"from boltzgen.{old_module}", f"from boltzgen_flat.{new_module}"),
        ]
        
        for old_pattern, new_pattern in patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                changes_made.append(f"{old_pattern} -> {new_pattern}")
    
    # Also handle import statements without 'from'
    for old_module, new_module in import_mapping.items():
        patterns = [
            (f"import boltzgen.src.boltzgen.{old_module}", f"import boltzgen_flat.{new_module}"),
            (f"import boltzgen.{old_module}", f"import boltzgen_flat.{new_module}"),
        ]
        
        for old_pattern, new_pattern in patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                changes_made.append(f"{old_pattern} -> {new_pattern}")
    
    # Handle module-level imports like "from boltzgen.data import const"
    # These need special handling since const is a module, not a file
    module_imports = [
        (r'from boltzgen\.src\.boltzgen\.data import const', 'from boltzgen_flat import data_const as const'),
        (r'from boltzgen\.data import const', 'from boltzgen_flat import data_const as const'),
        (r'import boltzgen\.src\.boltzgen\.data\.const', 'from boltzgen_flat import data_const as const'),
        (r'import boltzgen\.data\.const', 'from boltzgen_flat import data_const as const'),
    ]
    
    for pattern, replacement in module_imports:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes_made.append(f"{pattern} -> {replacement}")
    
    # Handle "from boltzgen.X.Y import Z as alias" patterns
    # This is trickier because we need to map the module path to the flat name
    import_as_pattern = r'from boltzgen(?:\.src\.boltzgen)?\.(\S+) import (\S+) as (\S+)'
    
    def replace_import_as(match):
        module_path = match.group(1)  # e.g., "model.layers"
        imported_name = match.group(2)  # e.g., "initialize"
        alias = match.group(3)  # e.g., "init"
        
        # Build the full module path
        full_module = module_path.replace('.', '_')
        
        # Try to find a matching flat module
        for old_mod, new_mod in import_mapping.items():
            if old_mod.endswith(imported_name) or old_mod == module_path:
                return f"from boltzgen_flat import {new_mod} as {alias}"
            # Check if it's a submodule import
            if module_path in old_mod:
                # This is importing a specific item from a module
                # e.g., from boltzgen.model.layers import initialize
                # should become: from boltzgen_flat import model_layers_initialize as init
                potential_flat = module_path.replace('.', '_') + '_' + imported_name
                if potential_flat in import_mapping.values():
                    return f"from boltzgen_flat import {potential_flat} as {alias}"
        
        # If no match found, just replace boltzgen with boltzgen_flat
        return f"from boltzgen_flat.{full_module} import {imported_name} as {alias}"
    
    content = re.sub(import_as_pattern, replace_import_as, content)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes_made
    
    return False, []

def fix_all_flat_imports(flat_dir, import_mapping):
    """Fix imports in all Python files in the boltzgen_flat directory."""
    python_files = list(flat_dir.glob('*.py'))
    
    updated_count = 0
    total_changes = 0
    
    print(f"\nProcessing {len(python_files)} files in {flat_dir}...")
    
    for file_path in python_files:
        updated, changes = fix_imports_in_flat_file(file_path, import_mapping)
        if updated:
            updated_count += 1
            total_changes += len(changes)
            print(f"\n[OK] Updated: {file_path.name}")
            if len(changes) <= 5:
                for change in changes:
                    print(f"    {change}")
            else:
                print(f"    ({len(changes)} changes made)")
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Files updated: {updated_count}/{len(python_files)}")
    print(f"  Total changes: {total_changes}")
    print(f"{'='*70}")
    
    return updated_count

def main():
    """Main function."""
    print("=" * 70)
    print("Fix Internal Imports in boltzgen_flat")
    print("=" * 70)
    
    # Check if flat directory exists
    if not BOLTZGEN_FLAT_DIR.exists():
        print(f"ERROR: Directory not found: {BOLTZGEN_FLAT_DIR}")
        return
    
    # Load import mapping
    print("\nLoading import mapping...")
    import_mapping = load_import_mapping()
    if import_mapping is None:
        return
    
    print(f"Loaded {len(import_mapping)} import mappings")
    
    # Fix imports
    print("\n" + "=" * 70)
    print("Fixing imports in boltzgen_flat files...")
    print("=" * 70)
    
    updated_count = fix_all_flat_imports(BOLTZGEN_FLAT_DIR, import_mapping)
    
    # Final message
    print("\n" + "=" * 70)
    print("Import fixing complete!")
    print("=" * 70)
    print(f"\nUpdated {updated_count} files in {BOLTZGEN_FLAT_DIR}")
    print("\nYou can now import from boltzgen_flat without needing the boltzgen module!")
    print("=" * 70)

if __name__ == '__main__':
    main()
