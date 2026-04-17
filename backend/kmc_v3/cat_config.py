"""
Physical catalyst geometry configuration using Cartesian coordinates
and explicit adsorption site types (atop, hollow, bridge).

Site assignment (corrected):
    ATOP   -> Carbon adsorption
    HOLLOW -> Hydrogen (stateless, counted via Langmuir theta_H)
    BRIDGE -> None (no adsorption)

Occupancy state is NOT stored in Site objects.
It lives in BaseKineticMC as 1D numpy arrays indexed by site index:
    occupancy[site_idx]      : 0=vacant, 1=single M-C, 2=double M-C (dMC)
    chain_at_site[site_idx]  : chain length of adsorbed carbon (0 if vacant)
    carbon_at_site[site_idx] : carbon array index (-1 if vacant) -> which carbon is adsorbed at this site
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json
from enum import Enum

#Define Enum for SiteType and AdsorbateType
class SiteType(Enum):
    ATOP   = "atop"    # On top of metal atom   -> Carbon
    HOLLOW = "hollow"  # In hollow between atoms -> Hydrogen (stateless)
    BRIDGE = "bridge"  # Between 2 metal atoms   -> None


class AdsorbateType(Enum):
    CARBON   = "C"
    HYDROGEN = "H"
    NONE     = "none"


@dataclass
class Site:
    """
    Acting as a struct for a single point on the surface.
    It stores the Cartesian position, 
              the type of site (atop/hollow/bridge), 
              and the type of adsorbate it can hold (C/H/none).
    *it doesn't store occupancy state* ; that lives in BaseKineticMC as separate arrays.
    """
    position:       np.ndarray
    site_type:      SiteType
    adsorbate_type: AdsorbateType


@dataclass
class SurfaceGeometry:
    """
    Defines the physical geometry of the catalyst surface.
    Followings are the placeholder for default values for Pt(111) — can be overridden by config files.
    """
    metal:            str               = "Pt"
    facet:            str               = "111"
    lattice_constant: float             = 3.92
    dimensions:       Tuple[int, int]   = (10, 10)
    periodic:         Tuple[bool, bool] = (True, True)


@dataclass
class CatalystConfig:
    """
    High-level configuration for combining the geometry with simulation rules.
    """
    geometry:        SurfaceGeometry
    c_site_type:     SiteType = SiteType.ATOP    # Carbon  -> atop
    h_site_type:     SiteType = SiteType.HOLLOW  # Hydrogen -> hollow (stateless)
    neighbor_cutoff: float    = 4.0

    @classmethod
    # Factory method: dictionary -> CatalystConfig
    def from_dict(cls, config: Dict) -> "CatalystConfig":
        return cls(
            geometry=SurfaceGeometry(**config.get("geometry", {})), #**unpack dictionary "geometry" and passes its keys as arguments
            c_site_type=SiteType(config.get("c_site_type", "atop")), #converts string to SiteType enum, default is "atop"
            h_site_type=SiteType(config.get("h_site_type", "hollow")), 
            neighbor_cutoff=config.get("neighbor_cutoff", 4.0), # the nearest neighbor distance on fcc111 is ~3.92, so 4.0 should capture nearest neighbors
        )

    @classmethod
    # Factory method: JSON file -> CatalystConfig 
    def from_file(cls, filepath: str) -> "CatalystConfig":
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))
        
    # Exporter: CatalystConfig -> JSON file   
    def to_file(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({
                "geometry": {
                    "metal":            self.geometry.metal,
                    "facet":            self.geometry.facet,
                    "lattice_constant": self.geometry.lattice_constant,
                    "dimensions":       list(self.geometry.dimensions),
                    "periodic":         list(self.geometry.periodic),
                },
                "c_site_type":     self.c_site_type.value,
                "h_site_type":     self.h_site_type.value,
                "neighbor_cutoff": self.neighbor_cutoff,
            }, f, indent=2)


class CatalystSurface:
    """
    Builds the physical surface geometry and pre-computes neighbor maps.

    Neighbor maps (computed once at __init__):
        c_neighbors[site_idx] : adjacent C site indices
                                -> used by dMC to find vacant partner site
        h_neighbors[site_idx] : adjacent H site indices
                                -> available for hydrogen counting if needed
    """

    def __init__(self, config: CatalystConfig):
        self.config          = config
        self.sites:          List[Site]       = []
        self.c_site_indices: List[int]        = []
        self.h_site_indices: List[int]        = []
        self.c_neighbors:    Dict[int, List[int]] = {}
        self.h_neighbors:    Dict[int, List[int]] = {}

        self._build_surface()
        self._build_neighbor_maps()

    # ------------------------------------------------------------------
    # Surface construction
    # ------------------------------------------------------------------

    def _build_surface(self):
        facet = self.config.geometry.facet
        if facet == "111":
            self._build_fcc111()
        elif facet == "100":
            self._build_fcc100()
        else:
            raise NotImplementedError(f"Facet ({facet}) not implemented")

        # Classify site indices into C/H
        for idx, site in enumerate(self.sites):
            if site.adsorbate_type == AdsorbateType.CARBON:
                self.c_site_indices.append(idx) # after building the sites list -> we classify each idx 
            elif site.adsorbate_type == AdsorbateType.HYDROGEN:
                self.h_site_indices.append(idx)

    def _build_fcc111(self):
        """
        Total number of sites = 2 * nx * ny (nx*ny C sites + nx*ny H sites)
        FCC(111) hexagonal surface.

        Lattice vectors (from slides):
            a1 = [a/2, a*sqrt(3)/2, 0]
            a2 = [a,   0,           0]

        ATOP   position : i*a1 + j*a2               -> Carbon
        HOLLOW position : i*a1 + j*a2 + (a1+a2)/3   -> Hydrogen (stateless)
        """
        a       = self.config.geometry.lattice_constant
        nx, ny  = self.config.geometry.dimensions
        a1      = np.array([a/2, a*np.sqrt(3)/2, 0.0])
        a2      = np.array([a, 0.0, 0.0])

        for i in range(nx):
            for j in range(ny):
                pos = i * a1 + j * a2
                self.sites.append(Site(pos.copy(), SiteType.ATOP, AdsorbateType.CARBON))
        #self.sites = [Site(position=[0.00, 0.00, 0], 
        #              site_type=SiteType.ATOP,   
        #              adsorbate_type=AdsorbateType.CARBON), ...  # idx 0 ~ n_c-1
        #total: nx * ny = n_c appends
        for i in range(nx):
            for j in range(ny):
                pos = i * a1 + j * a2 + (a1 + a2) / 3
                self.sites.append(Site(pos.copy(), SiteType.HOLLOW, AdsorbateType.HYDROGEN))
        #self.sites = [Site(position=[1.31, 1.13, 0], 
        #              site_type=SiteType.HOLLOW, 
        #              adsorbate_type=AdsorbateType.HYDROGEN), # idx n_c ~ n_c + n_h - 1
        # depends on the facet: for fcc111, n_h = n_c; for fcc100, n_h = (nx-1)*(ny-1)
        
    def _build_fcc100(self):
        """
        FCC(100) square surface.

        ATOP   position : (i*a,       j*a,       0) -> Carbon
        HOLLOW position : ((i+0.5)*a, (j+0.5)*a, 0) -> Hydrogen (stateless)
        """
        a      = self.config.geometry.lattice_constant
        nx, ny = self.config.geometry.dimensions

        for i in range(nx):
            for j in range(ny):
                pos = np.array([i * a, j * a, 0.0])
                self.sites.append(Site(pos.copy(), SiteType.ATOP, AdsorbateType.CARBON))

        for i in range(nx - 1):
            for j in range(ny - 1):
                pos = np.array([(i + 0.5) * a, (j + 0.5) * a, 0.0])
                self.sites.append(Site(pos.copy(), SiteType.HOLLOW, AdsorbateType.HYDROGEN))

    # ------------------------------------------------------------------
    # Neighbor map (run once)
    # ------------------------------------------------------------------

    def _build_neighbor_maps(self):
        """
        For every C site, find:
            c_neighbors : adjacent C sites  (dMC partner search)
            h_neighbors : adjacent H sites  (H counting if needed)
        """
        cutoff = self.config.neighbor_cutoff

        for c_idx in self.c_site_indices:
            c_pos  = self.sites[c_idx].position
            adj_c  = [oc for oc in self.c_site_indices
                      if oc != c_idx and self._distance(c_pos, self.sites[oc].position) <= cutoff]
            adj_h  = [h  for h  in self.h_site_indices
                      if self._distance(c_pos, self.sites[h].position)  <= cutoff]
            self.c_neighbors[c_idx] = adj_c
            self.h_neighbors[c_idx] = adj_h

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Minimum-image distance with periodic boundaries."""
        diff   = pos2 - pos1
        a      = self.config.geometry.lattice_constant
        nx, ny = self.config.geometry.dimensions
        if self.config.geometry.periodic[0]:
            Lx = nx * a
            diff[0] -= Lx * np.round(diff[0] / Lx)
        if self.config.geometry.periodic[1]:
            Ly = ny * a
            diff[1] -= Ly * np.round(diff[1] / Ly)
        return float(np.linalg.norm(diff))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_c_sites(self) -> int:
        return len(self.c_site_indices)

    @property
    def n_h_sites(self) -> int:
        return len(self.h_site_indices)

    def get_position(self, site_idx: int) -> np.ndarray:
        return self.sites[site_idx].position

    def get_c_neighbors(self, site_idx: int) -> List[int]:
        """Adjacent C sites (for dMC: find vacant neighbor)."""
        return self.c_neighbors.get(site_idx, [])

    def get_h_neighbors(self, site_idx: int) -> List[int]:
        """Adjacent H sites to a C site."""
        return self.h_neighbors.get(site_idx, [])

    def get_coordinates_array(self) -> np.ndarray:
        return np.array([s.position for s in self.sites])


# ------------------------------------------------------------------
# Predefined configurations
# ------------------------------------------------------------------

def pt111_config() -> CatalystConfig:
    """Pt(111) — typical hydrogenolysis catalyst"""
    return CatalystConfig(
        geometry=SurfaceGeometry(metal="Pt", facet="111",
                                 lattice_constant=3.92,
                                 dimensions=(20, 20), periodic=(True, True)),
        c_site_type=SiteType.ATOP, h_site_type=SiteType.HOLLOW, neighbor_cutoff=4.0)

def pt100_config() -> CatalystConfig:
    """Pt(100)"""
    return CatalystConfig(
        geometry=SurfaceGeometry(metal="Pt", facet="100",
                                 lattice_constant=3.92,
                                 dimensions=(20, 20), periodic=(True, True)),
        c_site_type=SiteType.ATOP, h_site_type=SiteType.HOLLOW, neighbor_cutoff=4.0)

def pd111_config() -> CatalystConfig:
    """Pd(111)"""
    return CatalystConfig(
        geometry=SurfaceGeometry(metal="Pd", facet="111",
                                 lattice_constant=3.89,
                                 dimensions=(20, 20), periodic=(True, True)),
        c_site_type=SiteType.ATOP, h_site_type=SiteType.HOLLOW, neighbor_cutoff=4.0)


example_json = """
{
  "geometry": {
    "metal": "Pt",
    "facet": "111",
    "lattice_constant": 3.92,
    "dimensions": [20, 20],
    "periodic": [true, true]
  },
  "c_site_type": "atop",
  "h_site_type": "hollow",
  "neighbor_cutoff": 4.0
}
"""