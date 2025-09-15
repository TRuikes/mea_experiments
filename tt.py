import subprocess
from pathlib import Path

def get_first_layer_folder_sizes(path="."):
    path = Path(path).resolve()
    result = subprocess.run(
        ["bash", "-c", f'du -s --block-size=1 "{path}"/*/'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    sizes = []
    for line in result.stdout.strip().split("\n"):
        if line:
            size, folder = line.strip().split("\t")
            sizes.append((Path(folder).name, int(size)))

    sizes.sort(key=lambda x: x[1], reverse=True)
    for name, size in sizes:
        print(f"{name:<30} {format_size(size)}")

def format_size(size_bytes):
    for unit in ['B','KB','MB','GB','TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

if __name__ == "__main__":
    get_first_layer_folder_sizes(".")
