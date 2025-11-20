#!/usr/bin/env python3
"""
Migration script to move all nested .py files from boltzgen to boltzgen_automodel/boltzgen_flat
and fix all imports to reference the new flat structure.
"""

import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
BOLTZGEN_DIR = Path("boltzgen/src/boltzgen")
TARGET_DIR = Path("boltzgen_automodel/boltzgen_flat")
AUTOMODEL_DIR = Path("boltzgen_automodel")
MAPPING_FILE = Path("migration_mapping.json")

def get_all_python_files(source_dir):
    """Recursively find all .py files in the source directory."""
    python_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def generate_flat_name(file_path, source_dir):
    """
    Generate a flat filename from a nested path.
    Skip __init__.py files.
    For conflicts, use parent directory names as prefixes.
    """
    # Skip __init__.py files
    if file_path.name == '__init__.py':
        return None
    
    # Get the relative path from source_dir
    rel_path = file_path.relative_to(source_dir)
    
    # If it's directly in the source dir, keep the name
    if len(rel_path.parts) == 1:
        return rel_path.name
    
    # Otherwise, create a name with parent directories as prefix
    # e.g., data/filter/dynamic/filter.py -> data_filter_dynamic_filter.py
    parts = list(rel_path.parts[:-1])  # All directories
    filename = rel_path.stem  # Filename without extension
    
    # Build the new name
    new_name = '_'.join(parts + [filename]) + '.py'
    return new_name

def create_file_mapping(source_dir, target_dir):
    """
    Create a mapping of source files to target flat filenames.
    Returns: dict mapping old_path -> new_flat_name
    """
    python_files = get_all_python_files(source_dir)
    mapping = {}
    name_counts = defaultdict(int)
    
    for file_path in python_files:
        flat_name = generate_flat_name(file_path, source_dir)
        
        # Skip __init__.py files
        if flat_name is None:
            continue
        
        # Track if we have duplicates (shouldn't happen with our naming scheme)
        name_counts[flat_name] += 1
        if name_counts[flat_name] > 1:
            print(f"WARNING: Duplicate flat name detected: {flat_name}")
        
        mapping[str(file_path)] = flat_name
    
    return mapping

def copy_files(mapping, target_dir):
    """Copy files to the target directory using the mapping."""
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for source_path, flat_name in mapping.items():
        source = Path(source_path)
        target = target_dir / flat_name
        
        # Copy the file
        shutil.copy2(source, target)
        copied_count += 1
        print(f"Copied: {source.name} -> {flat_name}")
    
    print(f"\nTotal files copied: {copied_count}")
    return copied_count

def create_import_mapping(file_mapping, source_dir):
    """
    Create a mapping of old import paths to new import paths.
    Returns: dict mapping old_import_path -> new_import_path
    """
    import_mapping = {}
    
    for source_path, flat_name in file_mapping.items():
        source = Path(source_path)
        
        # Get the module path relative to source_dir
        rel_path = source.relative_to(source_dir)
        
        # Convert path to module notation (remove .py, replace / with .)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        old_module = '.'.join(module_parts)
        
        # New module is just the flat filename without .py
        new_module = flat_name[:-3]  # Remove .py extension
        
        # Store the mapping
        import_mapping[old_module] = new_module
    
    return import_mapping

def fix_imports_in_file(file_path, import_mapping):
    """
    Fix imports in a single file.
    Handles patterns like:
    - from boltzgen.src.boltzgen.X.Y.Z import ...
    - from boltzgen.X.Y.Z import ...
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Pattern 1: from boltzgen.src.boltzgen.X.Y.Z import ...
    # Pattern 2: from boltzgen.X.Y.Z import ...
    for old_module, new_module in import_mapping.items():
        # Try both patterns
        patterns = [
            (f"from boltzgen.src.boltzgen.{old_module}", f"from boltzgen_flat.{new_module}"),
            (f"from boltzgen.{old_module}", f"from boltzgen_flat.{new_module}"),
        ]
        
        for old_pattern, new_pattern in patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                changes_made.append(f"{old_pattern} -> {new_pattern}")
    
    # Also handle import statements without 'from'
    # e.g., import boltzgen.src.boltzgen.X.Y.Z
    for old_module, new_module in import_mapping.items():
        patterns = [
            (f"import boltzgen.src.boltzgen.{old_module}", f"import boltzgen_flat.{new_module}"),
            (f"import boltzgen.{old_module}", f"import boltzgen_flat.{new_module}"),
        ]
        
        for old_pattern, new_pattern in patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                changes_made.append(f"{old_pattern} -> {new_pattern}")
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nUpdated imports in: {file_path.name}")
        for change in changes_made:
            print(f"  - {change}")
        return True
    
    return False

def fix_all_imports(automodel_dir, import_mapping):
    """Fix imports in all Python files in the automodel directory."""
    python_files = list(automodel_dir.glob('*.py'))
    
    updated_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path, import_mapping):
            updated_count += 1
    
    print(f"\nTotal files with updated imports: {updated_count}")
    return updated_count

def save_mapping(file_mapping, import_mapping, output_file):
    """Save the mapping to a JSON file for reference."""
    mapping_data = {
        'file_mapping': file_mapping,
        'import_mapping': import_mapping,
        'total_files': len(file_mapping),
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\nMapping saved to: {output_file}")

def main():
    """Main migration function."""
    print("=" * 70)
    print("BoltzGen Migration Script")
    print("=" * 70)
    
    # Check if source directory exists
    if not BOLTZGEN_DIR.exists():
        print(f"ERROR: Source directory not found: {BOLTZGEN_DIR}")
        return
    
    print(f"\nSource directory: {BOLTZGEN_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Files to update: {AUTOMODEL_DIR}/*.py")
    
    # Step 1: Create file mapping
    print("\n" + "=" * 70)
    print("Step 1: Creating file mapping...")
    print("=" * 70)
    file_mapping = create_file_mapping(BOLTZGEN_DIR, TARGET_DIR)
    print(f"Found {len(file_mapping)} files to migrate (excluding __init__.py)")
    
    # Step 2: Copy files
    print("\n" + "=" * 70)
    print("Step 2: Copying files to flat structure...")
    print("=" * 70)
    copied_count = copy_files(file_mapping, TARGET_DIR)
    
    # Step 3: Create import mapping
    print("\n" + "=" * 70)
    print("Step 3: Creating import mapping...")
    print("=" * 70)
    import_mapping = create_import_mapping(file_mapping, BOLTZGEN_DIR)
    print(f"Created {len(import_mapping)} import mappings")
    
    # Step 4: Fix imports
    print("\n" + "=" * 70)
    print("Step 4: Fixing imports in automodel files...")
    print("=" * 70)
    updated_count = fix_all_imports(AUTOMODEL_DIR, import_mapping)
    
    # Step 5: Save mapping
    print("\n" + "=" * 70)
    print("Step 5: Saving mapping file...")
    print("=" * 70)
    save_mapping(file_mapping, import_mapping, MAPPING_FILE)
    
    # Summary
    print("\n" + "=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    print(f"Files copied: {copied_count}")
    print(f"Files updated: {updated_count}")
    print(f"Mapping file: {MAPPING_FILE}")
    print("\nNext steps:")
    print("1. Review the migration_mapping.json file")
    print("2. Test importing from boltzgen_flat")
    print("3. Run any existing tests")
    print("=" * 70)

if __name__ == '__main__':
    main()
