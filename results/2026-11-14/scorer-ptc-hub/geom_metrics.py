"""Deciding-metric geometry of area "CAD": volumetric IoU and Chamfer distance between two triangle meshes.

numpy only. This is the scoring engine of `cad.cadquery_items@0.1` and deliberately shares no code with the
training reward (cad_protocol.reward, which uses OCCT's mass properties and bounding boxes): volumes are integrated
along seeded vertical lines from the signed crossings of each line with the mesh triangles (entering through a
downward-facing triangle, leaving through an upward-facing one), so overlapping solids of a compound count once. The meshes are binary STL files written by the executor (OCCT
tessellation of the solid at a relative deflection of 0.002 of the bounding-box diagonal); README "Metric and
tier" discusses what is and is not shared with the reward.

Deterministic: every random draw comes from `numpy.random.default_rng(seed)` with the seed passed in (score_cad.py
derives it from the item id), and every reduction is in a fixed order.
"""

from __future__ import annotations

import numpy as np

IOU_COLUMNS = 16384
CHAMFER_POINTS = 4096
_CHUNK = 256


def read_stl(path: str) -> np.ndarray:
    """Triangles (n, 3, 3) float64 of a binary or ASCII STL file."""
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
        if 84 + 50 * n == len(data):
            rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=rec, count=n, offset=84)
            return arr["v"].astype(np.float64)
    text = data.decode("ascii", errors="strict")
    verts = [list(map(float, line.split()[1:4])) for line in text.splitlines() if line.strip().startswith("vertex")]
    if len(verts) % 3:
        raise ValueError(f"{path}: ASCII STL with {len(verts)} vertices")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def bounds(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = tri.reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def _crossings(tri: np.ndarray, cols: np.ndarray, tag: int):
    """Crossings of vertical lines (x, y) = cols[j] with the mesh: (column, z, d_a, d_b) with the +1/-1 entry/exit
    increment in slot `tag` (0 = mesh A, 1 = mesh B). Entering is crossing a downward-facing triangle."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]          # twice the signed XY area; sign = sign of normal z
    keep = np.abs(det) > 1e-14
    a, b, c, e1, e2, det = a[keep], b[keep], c[keep], e1[keep], e2[keep], det[keep]
    out_col, out_z, out_d = [], [], []
    for s in range(0, len(cols), _CHUNK):
        q = cols[s:s + _CHUNK, None, :] - a[None, :, :2]        # (C, T, 2)
        u = (q[..., 0] * e2[None, :, 1] - q[..., 1] * e2[None, :, 0]) / det[None]
        v = (e1[None, :, 0] * q[..., 1] - e1[None, :, 1] * q[..., 0]) / det[None]
        hit = (u > 0) & (v > 0) & (u + v < 1)
        ci, ti = np.nonzero(hit)
        uu, vv = u[ci, ti], v[ci, ti]
        z = a[ti, 2] + uu * (b[ti, 2] - a[ti, 2]) + vv * (c[ti, 2] - a[ti, 2])
        out_col.append(ci + s)
        out_z.append(z)
        out_d.append(np.where(det[ti] < 0, 1, -1))
    col = np.concatenate(out_col) if out_col else np.zeros(0, int)
    z = np.concatenate(out_z) if out_z else np.zeros(0)
    d = np.concatenate(out_d) if out_d else np.zeros(0, int)
    dd = np.zeros((len(d), 2), dtype=np.int64)
    dd[:, tag] = d
    return col, z, dd


def volumetric_iou(cand: np.ndarray, ref: np.ndarray, seed: int, n: int = IOU_COLUMNS) -> dict:
    """IoU by vertical-line integration: `n` seeded uniform columns in the union of both XY bounding boxes; along
    each column the inside intervals of both solids are exact (signed crossings of the triangles), so the only
    sampling error is over XY. Also returns each mesh's volume estimate (a consistency check against OCCT)."""
    lo = np.minimum(bounds(cand)[0], bounds(ref)[0])
    hi = np.maximum(bounds(cand)[1], bounds(ref)[1])
    span = np.maximum(hi - lo, 1e-9)
    rng = np.random.default_rng(seed)
    cols = lo[:2] + rng.random((n, 2)) * span[:2]
    ca, za, da = _crossings(cand, cols, 0)
    cb, zb, db = _crossings(ref, cols, 1)
    col = np.concatenate([ca, cb])
    z = np.concatenate([za, zb])
    d = np.concatenate([da, db])
    order = np.lexsort((z, col))
    col, z, d = col[order], z[order], d[order]
    # running counts within each column: cumulative sum minus the column's starting offset
    cum = np.cumsum(d, axis=0)
    first = np.ones(len(col), dtype=bool)
    first[1:] = col[1:] != col[:-1]
    start_idx = np.maximum.accumulate(np.where(first, np.arange(len(col)), 0))
    base = cum[start_idx] - d[start_idx]
    cnt = cum - base                                           # counts just above each crossing
    same_next = np.zeros(len(col), dtype=bool)
    same_next[:-1] = col[1:] == col[:-1]
    seg = np.zeros(len(col))
    seg[:-1] = z[1:] - z[:-1]
    seg = np.where(same_next, seg, 0.0)
    in_a, in_b = cnt[:, 0] > 0, cnt[:, 1] > 0
    inter = float((seg * (in_a & in_b)).sum())
    union = float((seg * (in_a | in_b)).sum())
    col_area = float(span[0] * span[1]) / n
    unbalanced = int(np.count_nonzero(cnt[~same_next] != 0))   # a column must end outside both solids
    return {"iou": inter / union if union > 0 else 0.0, "n_columns": n,
            "cand_volume_mc": float((seg * in_a).sum()) * col_area,
            "ref_volume_mc": float((seg * in_b).sum()) * col_area,
            "unbalanced_columns": unbalanced}


def _surface_samples(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    p = area / area.sum()
    idx = rng.choice(len(tri), size=n, p=p)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a))
    for s in range(0, len(a), 512):
        d = ((a[s:s + 512, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[s:s + 512] = np.sqrt(d.min(axis=1))
    return out


def chamfer(cand: np.ndarray, ref: np.ndarray, seed: int, n: int = CHAMFER_POINTS) -> float:
    """Symmetric mean surface distance, divided by the reference bounding-box diagonal."""
    rng = np.random.default_rng(seed)
    pc, pr = _surface_samples(cand, n, rng), _surface_samples(ref, n, rng)
    lo, hi = bounds(ref)
    diag = float(np.linalg.norm(hi - lo))
    return float((_nn_dist(pc, pr).mean() + _nn_dist(pr, pc).mean()) / 2.0 / max(diag, 1e-9))
