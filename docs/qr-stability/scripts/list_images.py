from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
files=sorted(root.rglob("*.bmp"))
Path(sys.argv[2]).write_text("\n".join(map(str,files)),encoding="utf-8")
print(f"{len(files)} BMP images")
