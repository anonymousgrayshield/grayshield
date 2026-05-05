from __future__ import annotations
from typing import List, Dict

def pareto_front(points: List[Dict], x_key: str, y_key: str, maximize_y: bool = True) -> List[Dict]:
    pts = sorted(points, key=lambda d: d[x_key])
    front: List[Dict] = []
    best_y = None
    for p in pts:
        y = p[y_key]
        if best_y is None:
            front.append(p)
            best_y = y
            continue
        if maximize_y:
            if y > best_y:
                front.append(p)
                best_y = y
        else:
            if y < best_y:
                front.append(p)
                best_y = y
    return front
