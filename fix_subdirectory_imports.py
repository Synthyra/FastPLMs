#!/usr/bin/env python3
"""
Fix subdirectory imports in boltzgen_flat.

This script fixes imports like:
  from boltzgen_flat.model_layers_triangular_attention.attention import X
to:
  from boltzgen_flat.model_layers_triangular_attention_attention import X
"""

import re
from pathlib import Path

BOLTZGEN_FLAT_DIR = Path("boltzgen_automodel/boltzgen_flat")

def fix_subdirectory_imports(file_path):
    """Fix imports that reference subdirectories in the flat structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Pattern: from boltzgen_flat.X.Y import Z
    # Should be: from boltzgen_flat.X_Y import Z
    pattern = r'from boltzgen_flat\.(\S+)\.(\S+) import'
    
    def replace_subdirectory(match):
        part1 = match.group(1)  # e.g., "model_layers_triangular_attention"
        part2 = match.group(2)  # e.g., "attention"
        
        # Combine with underscore
        flat_name = f"{part1}_{part2}"
        return f"from boltzgen_flat.{flat_name} import"
    
    content = re.sub(pattern, replace_subdirectory, content)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """Main function."""
    print("=" * 70)
    print("Fix Subdirectory Imports in boltzgen_flat")
    print("=" * 70)
    
    python_files = list(BOLTZGEN_FLAT_DIR.glob('*.py'))
    updated_count = 0
    
    print(f"\nProcessing {len(python_files)} files...")
    
    for file_path in python_files:
        if fix_subdirectory_imports(file_path):
            updated_count += 1
            print(f"[OK] Updated: {file_path.name}")
    
    print(f"\n{'='*70}")
    print(f"Summary: Updated {updated_count}/{len(python_files)} files")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
