import os

IGNORE_DIRS = {"venv", ".venv","__pycache__", ".git", ".idea", ".vscode"}

def generate_tree(start_path, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(start_path):
            
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            level = root.replace(start_path, "").count(os.sep)
            indent = "│   " * level
            folder_name = os.path.basename(root) if root != start_path else start_path
            
            f.write(f"{indent}├── {folder_name}/\n")
            
            sub_indent = "│   " * (level + 1)
            for file in files:
                f.write(f"{sub_indent}├── {file}\n")

# Run
generate_tree(".", "structure.txt")
print("✅ Tree structure saved to structure.txt")