"""Check basic metadata in generated embedded model headers."""

from pathlib import Path
import re

HEADERS = [
    Path("../embedded_models/biofloc_compact_models.h"),
    Path("../embedded_models/fish_compact_models.h"),
]


def read_define(text: str, name: str):
    match = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", text)
    return int(match.group(1)) if match else None


def main():
    for header in HEADERS:
        text = header.read_text(encoding="utf-8", errors="replace")
        print(f"\n{header}")
        for define_name in ["BIOFLOC_N_FEATURES", "FISH_N_FEATURES"]:
            value = read_define(text, define_name)
            if value is not None:
                print(f"  {define_name}: {value}")
        for func in re.findall(r"float\s+(predict_[a-z0-9_]+)\s*\(", text):
            print(f"  prediction function: {func}")


if __name__ == "__main__":
    main()
