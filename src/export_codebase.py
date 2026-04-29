import os

OUTPUT_FILE = "codebase_dump.txt"

# folders to ignore (VERY IMPORTANT)
IGNORE_DIRS = {
    "venv", "__pycache__", ".git",
    "data", "csv_data",
    "faiss_index", "demo_faiss_index",
    "assets", "docs", "examples",
    ".agent", ".agents", ".gemini", ".gsd",
    "nl2sql"
}

def should_ignore(path):
    return any(ignored in path for ignored in IGNORE_DIRS)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk("."):
        # remove ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                if should_ignore(file_path):
                    continue

                out.write("="*80 + "\n")
                out.write(f"FILE: {file_path}\n")
                out.write("="*80 + "\n\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"[ERROR READING FILE: {e}]")

                out.write("\n\n\n")

print(f"✅ Done! Output saved in {OUTPUT_FILE}")