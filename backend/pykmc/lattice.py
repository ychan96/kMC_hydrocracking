"""
Catalyst surface geometry for the kMC framework.

Design
------
The simulation does NOT track site coordinates. Adsorbate diffusion is assumed
fast relative to chemistry, so site occupancy is uncorrelated and every
geometric effect reduces to a small set of integers. This module supplies those
integers; nothing spatial survives into the event loop.

Site model
----------
    carbon    -> BRIDGE site   (2 M-C bonds per carbon)
    hydrogen  -> HOLLOW site   (3-fold on hexagonal, 4-fold on square)
    An adsorbed C-C pair blocks `m` = 4 metal atoms: two disjoint bridges,
    4 M-C bonds, no shared metal atom.

Occupancy unit
--------------
theta is counted in METAL ATOMS, not bridge sites. Bridges share metal atoms,
so counting bridges would double-count blocking. Each adsorbed carbon blocks
2 metal atoms; each adsorbed C-C pair blocks `m`.

Consumed by the rate expressions as:
    k_scission = k0 * (1 - theta_C)**m
                    * (1 - theta_H + theta_H*exp(-w/kT))**N_H
    k_ads      = k0 * (1 - theta_C)**2
"""

from dataclasses import dataclass, replace
from typing import Dict, Tuple
import json
import numpy as np


# ----------------------------------------------------------------------
# Facet constants
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Facet:
    """
    Geometric constants for one surface termination.

    Frozen: these are fixed properties of the lattice, not tunable parameters.
    N_H in particular must NOT be fitted -- it is degenerate with the lateral
    interaction energy w, so it is frozen here and only w is fitted.

    z                 : in-plane metal coordination number
    m                 : metal atoms blocked by one adsorbed C-C pair
    bridges_per_atom  : bridge sites per surface metal atom  (= z / 2)
    hollows_per_atom  : hollow sites per surface metal atom
    N_H               : hollow sites adjacent to the adsorbed C-C pair,
                        i.e. the exponent in the lateral-H penalty
    """
    name:             str
    z:                int
    m:                int
    bridges_per_atom: float
    hollows_per_atom: float
    N_H:              int


# fcc(111) and hcp(0001) share an identical surface layer (2D hexagonal);
# they differ only in the stacking beneath it, which this model never sees.
HEXAGONAL = Facet(name="hexagonal", z=6, m=4,
                  bridges_per_atom=3.0, hollows_per_atom=2.0, N_H=2)

SQUARE    = Facet(name="square",    z=4, m=4,
                  bridges_per_atom=2.0, hollows_per_atom=1.0, N_H=1)

FACETS: Dict[str, Facet] = {
    "111":  HEXAGONAL,
    "0001": HEXAGONAL,
    "100":  SQUARE,
}


# ----------------------------------------------------------------------
# Surface geometry
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceGeometry:
    """
    Physical description of the catalyst surface patch.

    nn_distance is the SURFACE nearest-neighbour metal-metal spacing, not the
    bulk cubic lattice constant. For an fcc metal use `from_cubic()`; for hcp
    the basal a-parameter is already the surface spacing.

        Pt  a_cubic = 3.92 A  ->  nn = 2.77 A
        Pd  a_cubic = 3.89 A  ->  nn = 2.75 A
        Ru  a_hcp   = 2.71 A  ->  nn = 2.71 A
    """
    metal:       str                = "Ru"
    facet:       str                = "0001"
    nn_distance: float              = 2.71          # Angstrom
    dimensions:  Tuple[int, int]    = (20, 20)

    @classmethod
    def from_cubic(cls, metal: str, facet: str, a_cubic: float,
                   dimensions: Tuple[int, int] = (20, 20)) -> "SurfaceGeometry":
        """Build from an fcc cubic lattice constant (nn = a / sqrt(2))."""
        return cls(metal=metal, facet=facet,
                   nn_distance=a_cubic / np.sqrt(2.0),
                   dimensions=dimensions)


# ----------------------------------------------------------------------
# Configuration consumed by the simulation
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CatalystConfig:
    """
    Everything the kMC run needs to know about the surface.

    All site counts are integers derived once from the geometry. The event loop
    reads n_metal and the Facet fields; it never touches coordinates.
    """
    geometry: SurfaceGeometry

    @property
    def facet_constants(self) -> Facet:
        try:
            return FACETS[self.geometry.facet]
        except KeyError:
            raise NotImplementedError(
                f"facet {self.geometry.facet!r} not implemented; "
                f"available: {sorted(FACETS)}"
            ) from None

    # -- site inventory ------------------------------------------------

    @property
    def n_metal(self) -> int:
        """Surface metal atoms. This is the denominator of theta_C."""
        nx, ny = self.geometry.dimensions
        return nx * ny

    @property
    def n_bridge(self) -> int:
        """Bridge sites (carbon adsorption sites)."""
        return int(round(self.facet_constants.bridges_per_atom * self.n_metal))

    @property
    def n_hollow(self) -> int:
        """Hollow sites (hydrogen adsorption sites)."""
        return int(round(self.facet_constants.hollows_per_atom * self.n_metal))

    @property
    def area(self) -> float:
        """Patch area in A^2 (for turnover normalisation)."""
        d = self.geometry.nn_distance
        nx, ny = self.geometry.dimensions
        cell = d * d * (np.sqrt(3) / 2 if self.facet_constants.z == 6 else 1.0)
        return float(nx * ny * cell)

    # -- convenience ---------------------------------------------------

    @property
    def m(self) -> int:
        """Metal atoms blocked per adsorbed C-C pair."""
        return self.facet_constants.m

    @property
    def N_H(self) -> int:
        """Hollow sites adjacent to the C-C pair (lateral-H penalty exponent)."""
        return self.facet_constants.N_H

    def summary(self) -> str:
        f = self.facet_constants
        g = self.geometry
        return (
            f"{g.metal}({g.facet}) {g.dimensions[0]}x{g.dimensions[1]}  "
            f"[{f.name}]\n"
            f"  nn spacing      : {g.nn_distance:.3f} A\n"
            f"  metal atoms     : {self.n_metal}\n"
            f"  bridge sites    : {self.n_bridge}   (carbon)\n"
            f"  hollow sites    : {self.n_hollow}   (hydrogen)\n"
            f"  z               : {f.z}\n"
            f"  m  (blocked/CC) : {f.m}\n"
            f"  N_H (penalty)   : {f.N_H}\n"
            f"  area            : {self.area:.1f} A^2"
        )

    # -- serialisation -------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict) -> "CatalystConfig":
        g = d.get("geometry", {})
        return cls(geometry=SurfaceGeometry(
            metal=g.get("metal", "Ru"),
            facet=g.get("facet", "0001"),
            nn_distance=g.get("nn_distance", 2.71),
            dimensions=tuple(g.get("dimensions", (20, 20))),
        ))

    @classmethod
    def from_file(cls, filepath: str) -> "CatalystConfig":
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict:
        g = self.geometry
        return {"geometry": {"metal": g.metal, "facet": g.facet,
                             "nn_distance": g.nn_distance,
                             "dimensions": list(g.dimensions)}}

    def to_file(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

def ru0001_config(dimensions: Tuple[int, int] = (20, 20)) -> CatalystConfig:
    """Ru(0001) -- hexagonal. a_hcp is already the surface spacing."""
    return CatalystConfig(SurfaceGeometry("Ru", "0001", 2.71, dimensions))


def pt111_config(dimensions: Tuple[int, int] = (20, 20)) -> CatalystConfig:
    """Pt(111) -- hexagonal, same surface layer as hcp(0001)."""
    return CatalystConfig(SurfaceGeometry.from_cubic("Pt", "111", 3.92, dimensions))


def pt100_config(dimensions: Tuple[int, int] = (20, 20)) -> CatalystConfig:
    """Pt(100) -- square."""
    return CatalystConfig(SurfaceGeometry.from_cubic("Pt", "100", 3.92, dimensions))


# ----------------------------------------------------------------------
# Offline verification of the tabulated constants
# ----------------------------------------------------------------------

def verify_site_ratios(n: int = 40) -> Dict[str, Dict[str, float]]:
    """
    Rebuild each lattice explicitly and confirm bridges_per_atom and
    hollows_per_atom converge to the tabulated values.

    Never called at runtime -- this exists so the constants above are
    reproducible rather than asserted. Run it once; hardcode the result.
    """
    out = {}

    # hexagonal: triangular metal net, hollows = triangle centres (fcc + hcp)
    hex_nb = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]),
                        (p[0], p[1] + 1), (p[0], p[1] - 1),
                        (p[0] + 1, p[1] - 1), (p[0] - 1, p[1] + 1)]
    atoms = {(i, j) for i in range(n) for j in range(n)}
    bridges = {frozenset((p, q)) for p in atoms for q in hex_nb(p) if q in atoms}
    hollows = {frozenset((p, q, r)) for p in atoms for q in hex_nb(p)
               for r in hex_nb(p) if q in atoms and r in atoms and r in hex_nb(q)}
    out["hexagonal"] = {"bridges_per_atom": len(bridges) / len(atoms),
                        "hollows_per_atom": len(hollows) / len(atoms)}

    # square: 4 in-plane neighbours, hollows = plaquette centres
    sq_nb = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]),
                       (p[0], p[1] + 1), (p[0], p[1] - 1)]
    bridges = {frozenset((p, q)) for p in atoms for q in sq_nb(p) if q in atoms}
    plaq = {frozenset(((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)))
            for i in range(n - 1) for j in range(n - 1)}
    out["square"] = {"bridges_per_atom": len(bridges) / len(atoms),
                     "hollows_per_atom": len(plaq) / len(atoms)}
    return out


if __name__ == "__main__":
    for cfg in (ru0001_config(), pt111_config(), pt100_config()):
        print(cfg.summary()); print()
    print("site-ratio verification (finite patch, converges to tabulated):")
    for facet, r in verify_site_ratios().items():
        print(f"  {facet:10s} {r}")