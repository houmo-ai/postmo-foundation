#!/usr/bin/env python3
"""
Script to add Apache 2.0 license headers to Python source files.
"""

import os
from pathlib import Path

LICENSE_HEADER = """# Copyright 2024-2026 HLIEPython Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""


def has_license_header(content: str) -> bool:
    """Check if file already has a license header."""
    return "Apache License" in content[:500] or "Copyright" in content[:200]


def add_header_to_file(file_path: Path) -> bool:
    """Add license header to a Python file if it doesn't have one."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if has_license_header(content):
            print(f"Skipping {file_path} - already has license header")
            return False

        # Preserve shebang if present
        lines = content.split('\n')
        new_content = LICENSE_HEADER

        if lines and lines[0].startswith('#!'):
            new_content = lines[0] + '\n' + LICENSE_HEADER + '\n'.join(lines[1:])
        else:
            new_content = LICENSE_HEADER + content

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"Added license header to {file_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Add license headers to all Python files in hliepython package."""
    base_path = Path(__file__).parent.parent / "hliepython"

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return

    python_files = list(base_path.rglob("*.py"))
    print(f"Found {len(python_files)} Python files")

    modified_count = 0
    for py_file in python_files:
        if add_header_to_file(py_file):
            modified_count += 1

    print(f"\nProcessed {len(python_files)} files, modified {modified_count} files")


if __name__ == "__main__":
    main()
