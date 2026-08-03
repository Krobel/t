"""
Template for exporting scikit-learn tree models to compact C arrays.

This is a starting point for reproducible export. It is not a replacement for the
validated generated headers already included in `embedded_models/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def tree_to_c_array(tree, array_name: str) -> str:
    """Convert one fitted sklearn decision tree into a compact C node array."""
    t = tree.tree_
    lines = [f"static const TreeNode {array_name}[] = {{"]
    for i in range(t.node_count):
        feature = int(t.feature[i])
        threshold = float(t.threshold[i])
        left = int(t.children_left[i])
        right = int(t.children_right[i])
        value = float(t.value[i].ravel()[0])
        if left == -1 and right == -1:
            feature = -1
            threshold = 0.0
        lines.append(
            f"    {{{feature}, {threshold:.9g}f, {left}, {right}, {value:.9g}f}},"
        )
    lines.append("};")
    return "\n".join(lines)


def write_header(arrays: Iterable[str], output_path: str | Path):
    output_path = Path(output_path)
    body = "\n\n".join(arrays)
    header = f"""#ifndef EXPORTED_TREE_MODELS_H
#define EXPORTED_TREE_MODELS_H

#include <stdint.h>

typedef struct {{
    int16_t feature;
    float threshold;
    int16_t left;
    int16_t right;
    float value;
}} TreeNode;

{body}

#endif
"""
    output_path.write_text(header, encoding="utf-8")


if __name__ == "__main__":
    print("Import this module from your trained-model script and call tree_to_c_array().")
