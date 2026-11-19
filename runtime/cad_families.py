"""Parametric part families of area "CAD": a text description and a reference CadQuery program per parameter draw.

Standard library only. Every description fixes the placement (what is centred where, which axis is which), because
the deciding metric (volumetric IoU, score_cad.py) compares solids in absolute coordinates. Dimensions are whole
millimetres (a few half millimetres). Each family has two phrasings; the phrasing index is part of the draw.

`draw(family, rng)` returns `(key, description, code)`; `key` is the canonical parameter tuple used for the
train/item disjointness rule (make_data.py): a key appears in at most one of SFT, GRPO and the items.
"""

from __future__ import annotations

import math
import random


def _f(x: float) -> str:
    """Number in text and code: integers without a decimal point, halves with one."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def box(r: random.Random):
    L, W, H = r.randint(10, 90), r.randint(10, 90), r.randint(3, 50)
    ph = r.randrange(2)
    desc = [f"A rectangular block {_f(L)} mm long in X, {_f(W)} mm wide in Y and {_f(H)} mm tall in Z, "
            f"centred at the origin.",
            f"A solid box measuring {_f(L)} x {_f(W)} x {_f(H)} mm (X x Y x Z) whose centre is at the origin."][ph]
    code = f"import cadquery as cq\n\nresult = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(H)})"
    return ("box", L, W, H, ph), desc, code


def cylinder(r: random.Random):
    D, H = r.randint(6, 90), r.randint(2, 90)
    ph = r.randrange(2)
    desc = [f"A solid cylinder of diameter {_f(D)} mm and height {_f(H)} mm with its axis along Z, centred at the "
            f"origin.",
            f"A round rod {_f(H)} mm long along the Z axis with a diameter of {_f(D)} mm; its centre is at the "
            f"origin."][ph]
    code = f"import cadquery as cq\n\nresult = cq.Workplane(\"XY\").cylinder({_f(H)}, {_f(D / 2)})"
    return ("cylinder", D, H, ph), desc, code


def tube(r: random.Random):
    Do = r.randint(10, 90)
    Di = r.randint(4, Do - 4)
    H = r.randint(4, 90)
    ph = r.randrange(2)
    desc = [f"A tube (hollow cylinder) with outer diameter {_f(Do)} mm, inner diameter {_f(Di)} mm and length "
            f"{_f(H)} mm along Z, centred at the origin.",
            f"A pipe section {_f(H)} mm long whose axis is the Z axis, outer diameter {_f(Do)} mm and bore "
            f"{_f(Di)} mm, centred at the origin."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").circle({_f(Do / 2)}).circle({_f(Di / 2)}).extrude({_f(H / 2)}, both=True)")
    return ("tube", Do, Di, H, ph), desc, code


def plate_hole(r: random.Random):
    L, W = r.randint(20, 100), r.randint(20, 100)
    T = r.randint(2, 15)
    D = r.randint(3, min(L, W) - 6)
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} mm plate, {_f(T)} mm thick in Z, centred at the origin, with a {_f(D)} mm "
            f"diameter through hole at its centre.",
            f"A flat plate {_f(L)} mm along X, {_f(W)} mm along Y and {_f(T)} mm thick, centred at the origin, "
            f"pierced by a central vertical hole of {_f(D)} mm diameter."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(T)}).faces(\">Z\").workplane().hole({_f(D)})")
    return ("plate_hole", L, W, T, D, ph), desc, code


def plate_4holes(r: random.Random):
    L, W = r.randint(30, 120), r.randint(30, 120)
    T = r.randint(2, 15)
    D = r.randint(3, 10)
    e = r.randint(math.ceil(D / 2) + 2, max(math.ceil(D / 2) + 2, min(L, W) // 4))
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} mm plate (X x Y), {_f(T)} mm thick, centred at the origin, with four {_f(D)} mm "
            f"through holes, one near each corner, each hole centre {_f(e)} mm in from both adjacent edges.",
            f"A rectangular mounting plate centred at the origin, {_f(L)} mm in X by {_f(W)} mm in Y and {_f(T)} mm "
            f"thick, with a {_f(D)} mm vertical through hole at each corner, centred {_f(e)} mm from the two "
            f"nearest edges."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = (\n    cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(T)})\n    .faces(\">Z\").workplane()\n"
            f"    .rect({_f(L - 2 * e)}, {_f(W - 2 * e)}, forConstruction=True).vertices()\n    .hole({_f(D)})\n)")
    return ("plate_4holes", L, W, T, D, e, ph), desc, code


def hex_prism(r: random.Random):
    F, H = r.randint(8, 70), r.randint(3, 60)
    ph = r.randrange(2)
    desc = [f"A hexagonal prism {_f(H)} mm tall along Z, centred at the origin, measuring {_f(F)} mm across flats, "
            f"with two of its flat sides parallel to the X axis.",
            f"A regular hexagonal bar, {_f(F)} mm across the flats and {_f(H)} mm long in Z, centred at the "
            f"origin, oriented so that one pair of flats is parallel to the X axis."][ph]
    d = F / math.cos(math.pi / 6)
    code = ("import cadquery as cq\n\n"
            f"across_flats = {_f(F)}\n"
            "circum_diameter = across_flats / 0.8660254037844386\n"
            f"result = cq.Workplane(\"XY\").polygon(6, circum_diameter).extrude({_f(H / 2)}, both=True)")
    del d
    return ("hex_prism", F, H, ph), desc, code


def stepped_shaft(r: random.Random):
    D1 = r.randint(12, 80)
    D2 = r.randint(5, D1 - 4)
    H1, H2 = r.randint(3, 50), r.randint(3, 60)
    ph = r.randrange(2)
    desc = [f"Two coaxial cylinders on the Z axis: a base cylinder of diameter {_f(D1)} mm from z = 0 to "
            f"z = {_f(H1)}, and on top of it a cylinder of diameter {_f(D2)} mm from z = {_f(H1)} to "
            f"z = {_f(H1 + H2)}.",
            f"A stepped shaft standing on the XY plane with its axis on Z: a {_f(D1)} mm diameter section "
            f"{_f(H1)} mm high at the bottom (starting at z = 0) and a {_f(D2)} mm diameter section {_f(H2)} mm "
            f"high directly above it."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = (\n    cq.Workplane(\"XY\").circle({_f(D1 / 2)}).extrude({_f(H1)})\n"
            f"    .faces(\">Z\").workplane().circle({_f(D2 / 2)}).extrude({_f(H2)})\n)")
    return ("stepped_shaft", D1, H1, D2, H2, ph), desc, code


def fillet_box(r: random.Random):
    L, W, H = r.randint(15, 90), r.randint(15, 90), r.randint(3, 40)
    R = r.randint(1, min(L, W) // 2 - 2)
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} x {_f(H)} mm block (X, Y, Z) centred at the origin, with its four vertical "
            f"edges rounded to a {_f(R)} mm radius.",
            f"A rounded-corner slab centred at the origin: {_f(L)} mm in X, {_f(W)} mm in Y, {_f(H)} mm in Z, "
            f"the four edges parallel to Z filleted with radius {_f(R)} mm."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(H)}).edges(\"|Z\").fillet({_f(R)})")
    return ("fillet_box", L, W, H, R, ph), desc, code


def chamfer_box(r: random.Random):
    L, W, H = r.randint(15, 90), r.randint(15, 90), r.randint(6, 40)
    C = r.randint(1, max(1, min(L, W, H) // 3))
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} x {_f(H)} mm block (X, Y, Z) centred at the origin, with the four edges of its "
            f"top face chamfered by {_f(C)} mm.",
            f"A box centred at the origin, {_f(L)} mm in X, {_f(W)} mm in Y and {_f(H)} mm in Z, whose top "
            f"(highest Z) face has a {_f(C)} mm x 45 degree chamfer on all four edges."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(H)}).edges(\">Z\").chamfer({_f(C)})")
    return ("chamfer_box", L, W, H, C, ph), desc, code


def cup(r: random.Random):
    L, W, H = r.randint(20, 90), r.randint(20, 90), r.randint(10, 60)
    t = r.randint(1, max(1, min(L, W, H) // 6))
    ph = r.randrange(2)
    desc = [f"An open-top rectangular box {_f(L)} x {_f(W)} x {_f(H)} mm outside (X, Y, Z), centred at the "
            f"origin, with {_f(t)} mm thick walls and floor; the top face is open.",
            f"A rectangular tray centred at the origin with outer size {_f(L)} mm (X) by {_f(W)} mm (Y) by "
            f"{_f(H)} mm (Z), hollowed out from the top so that the walls and the bottom are {_f(t)} mm thick."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(H)}).faces(\">Z\").shell(-{_f(t)})")
    return ("cup", L, W, H, t, ph), desc, code


def blind_hole(r: random.Random):
    D, H = r.randint(12, 80), r.randint(8, 80)
    d = r.randint(3, D - 6)
    depth = r.randint(2, H - 2)
    ph = r.randrange(2)
    desc = [f"A cylinder of diameter {_f(D)} mm and height {_f(H)} mm along Z, centred at the origin, with a "
            f"{_f(d)} mm diameter hole {_f(depth)} mm deep drilled into its top face along the axis.",
            f"A round boss centred at the origin, {_f(D)} mm in diameter and {_f(H)} mm tall in Z, with a "
            f"centred blind bore of {_f(d)} mm diameter going {_f(depth)} mm down from the top face."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").cylinder({_f(H)}, {_f(D / 2)}).faces(\">Z\").workplane()"
            f".hole({_f(d)}, {_f(depth)})")
    return ("blind_hole", D, H, d, depth, ph), desc, code


def slot_plate(r: random.Random):
    L, W, T = r.randint(30, 120), r.randint(20, 80), r.randint(2, 12)
    sw = r.randint(3, max(3, W // 3))
    sl = r.randint(sw + 2, max(sw + 2, L - 10))
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} x {_f(T)} mm plate (X, Y, Z) centred at the origin with a straight through slot "
            f"at its centre, {_f(sl)} mm long end to end along X and {_f(sw)} mm wide, with fully rounded ends.",
            f"A plate centred at the origin, {_f(L)} mm in X, {_f(W)} mm in Y and {_f(T)} mm thick, with a "
            f"central obround slot cut through it: overall length {_f(sl)} mm along X, width {_f(sw)} mm."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = (\n    cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(T)})\n"
            f"    .faces(\">Z\").workplane().slot2D({_f(sl)}, {_f(sw)}).cutThruAll()\n)")
    return ("slot_plate", L, W, T, sl, sw, ph), desc, code


def l_bracket(r: random.Random):
    L, W, T = r.randint(20, 90), r.randint(15, 70), r.randint(2, 10)
    H = r.randint(T + 8, 80)
    ph = r.randrange(2)
    desc = [f"An L-shaped bracket made of two plates {_f(L)} mm long in X and centred on X = 0: a base plate "
            f"{_f(W)} mm in Y (from y = -{_f(W / 2)} to y = {_f(W / 2)}) and {_f(T)} mm thick, from z = 0 to "
            f"z = {_f(T)}; and an upright plate {_f(T)} mm thick in Y, from y = -{_f(W / 2)} to "
            f"y = {_f(-W / 2 + T)}, rising from z = 0 to z = {_f(H)}.",
            f"An angle bracket: a horizontal plate {_f(L)} x {_f(W)} mm (X x Y) and {_f(T)} mm thick lying on the "
            f"XY plane (z from 0 to {_f(T)}), centred on the origin in X and Y, joined along its -Y edge to a "
            f"vertical plate of the same length and thickness that is {_f(H)} mm tall (z from 0 to {_f(H)}) and "
            f"spans y = -{_f(W / 2)} to y = {_f(-W / 2 + T)}."][ph]
    code = ("import cadquery as cq\n\n"
            f"base = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(T)}, centered=(True, True, False))\n"
            f"upright = (\n    cq.Workplane(\"XY\").box({_f(L)}, {_f(T)}, {_f(H)}, centered=(True, False, False))\n"
            f"    .translate((0, -{_f(W / 2)}, 0))\n)\n"
            "result = base.union(upright)")
    return ("l_bracket", L, W, T, H, ph), desc, code


def cbore_plate(r: random.Random):
    L, W, T = r.randint(25, 100), r.randint(25, 100), r.randint(5, 20)
    d = r.randint(3, 12)
    Dc = r.randint(d + 3, min(d + 12, min(L, W) - 6))
    cd = r.randint(1, T - 2)
    ph = r.randrange(2)
    desc = [f"A {_f(L)} x {_f(W)} x {_f(T)} mm plate (X, Y, Z) centred at the origin with a counterbored through "
            f"hole at its centre: hole diameter {_f(d)} mm, counterbore diameter {_f(Dc)} mm, counterbore depth "
            f"{_f(cd)} mm from the top face.",
            f"A plate centred at the origin, {_f(L)} mm in X, {_f(W)} mm in Y, {_f(T)} mm thick, with one central "
            f"vertical {_f(d)} mm through hole that is counterbored from the top (+Z) side to {_f(Dc)} mm "
            f"diameter and {_f(cd)} mm depth."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = cq.Workplane(\"XY\").box({_f(L)}, {_f(W)}, {_f(T)}).faces(\">Z\").workplane()"
            f".cboreHole({_f(d)}, {_f(Dc)}, {_f(cd)})")
    return ("cbore_plate", L, W, T, d, Dc, cd, ph), desc, code


def flange(r: random.Random):
    D = r.randint(30, 120)
    T = r.randint(3, 15)
    n = r.choice([3, 4, 5, 6, 8])
    d = r.randint(3, max(3, D // 8))
    pcd = r.randint(max(d + 6, D // 3), D - d - 4)
    ph = r.randrange(2)
    desc = [f"A circular disc of diameter {_f(D)} mm and thickness {_f(T)} mm along Z, centred at the origin, with "
            f"{n} through holes of {_f(d)} mm diameter equally spaced on a {_f(pcd)} mm diameter circle around the "
            f"axis, the first hole on the +X axis.",
            f"A flat round flange centred at the origin, {_f(D)} mm across and {_f(T)} mm thick (Z), with a bolt "
            f"circle of {_f(pcd)} mm diameter carrying {n} evenly spaced {_f(d)} mm holes through it, one of them "
            f"centred on the positive X axis."][ph]
    code = ("import cadquery as cq\n\n"
            f"result = (\n    cq.Workplane(\"XY\").cylinder({_f(T)}, {_f(D / 2)})\n"
            f"    .faces(\">Z\").workplane()\n    .polarArray({_f(pcd / 2)}, 0, 360, {n})\n    .hole({_f(d)})\n)")
    return ("flange", D, T, n, d, pcd, ph), desc, code


FAMILIES = {f.__name__: f for f in (box, cylinder, tube, plate_hole, plate_4holes, hex_prism, stepped_shaft,
                                     fillet_box, chamfer_box, cup, blind_hole, slot_plate, l_bracket, cbore_plate,
                                     flange)}


def draw(family: str, r: random.Random):
    return FAMILIES[family](r)


def geometry_key(key: tuple) -> tuple:
    """Key without the phrasing index: the disjointness rule is on the geometry, not only the wording."""
    return key[:-1]
