"""
Geometry 
----------------------------------------------------------------
    atop     one per metal atom        id = i * ny + j
    bridge   midpoint of a metal pair  id = atop_id * n_dirs  + d
    hollow   centroid of a 3/4-ring    id = atop_id * n_kinds + u

Occupancy model
---------------
1.  CARBON SITS ONLY ON BRIDGES. Hydrogen sits only on hollows.

2.  H IS STATIC.(quasi-equilibrium: H2 dissociation/recombination is fast
    enough)

3.  dMC PARTNERS are the bridges sharing EXACTLY ONE metal atom at the
    nearest such separation

Arrays the reaction layer uses directly
---------------------------------------
    occupancy[bid]      0 vacant | 1 single M-C | 2 dMC
    carbon_at_site[bid] absolute carbon index, or -1
    chain_at_site[bid]  +N single M-C on an N-mer | -N dMC pair | 0 vacant
    hollow_H[hid]       bool, frozen after construction
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Geometry 
# ══════════════════════════════════════════════════════════════════════════

BRIDGE_OFFSETS = {"hex":    [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 0), (0, 1))], #3 bridges per atom
                  "square": [((0, 0), (1, 0)), ((0, 0), (0, 1))]}                   #2 bridges per atom
HOLLOW_CORNERS = {"hex":    [[(0, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (1, 1)]],   #2 kinds, 3 corners each
                  "square": [[(0, 0), (1, 0), (0, 1), (1, 1)]]}                     #1 kind, 4 corners 

_kind = lambda f: "hex" if f in ("111", "0001") else "square"


def lattice_vectors(facet, a):
    if facet in ("111", "0001"):
        return np.array([a, 0.0]), np.array([a / 2, a * np.sqrt(3) / 2])
    if facet == "100":
        return np.array([a, 0.0]), np.array([0.0, a])
    raise NotImplementedError(f"facet {facet!r}")


def atop_positions(facet, nx, ny, a):
    a1, a2 = lattice_vectors(facet, a)
    i, j = np.divmod(np.arange(nx * ny), ny)
    # c = np.arrange(nx * ny) = [0, 1, 2, ..., nx*ny-1]
    # np.divmod(c, ny) = (i=c//ny, j=c%ny): row-major indexing
    return i[:, None] * a1 + j[:, None] * a2 # i[:, None]; row vector -> column vector (nx*ny, 1)


def bridge_positions(facet, nx, ny, a):
    a1, a2 = lattice_vectors(facet, a)
    offs = BRIDGE_OFFSETS[_kind(facet)]
    i, j = np.divmod(np.arange(nx * ny), ny)  
    out = np.empty((nx * ny * len(offs), 2)) #nx=3, ny=4, facet(111) -> len(offs)=3(bridges per atom) -> out.shape=(36, 2)
    for d, ((p0, q0), (p1, q1)) in enumerate(offs):
        r0 = (i + p0)[:, None] * a1 + (j + q0)[:, None] * a2
        r1 = (i + p1)[:, None] * a1 + (j + q1)[:, None] * a2
        out[d::len(offs)] = (r0 + r1) / 2 #midpoint of the two atop = bridge
    # i.e., facet(111)
    # offs = [((0,0),(1,0)), ((0,0),(0,1)), ((1,0),(0,1))]
    # list(enumerate(offs)) = [ (0, ((0,0),(1,0))), (1, ((0,0),(0,1))), (2, ((1,0),(0,1))) ]
    # (i,j) = cell coordinates
    
    # i.e., bid = 3 -> (i*ny+j)*n_dir + d -> (i,j,d) = (0,1,0) -> d=0 -> offs[0] = ((0,0),(1,0))
    # r0 = (i+0, j+0) = atop (0,1)
    # r1 = (i+1, j+0) = atop (1,1)
    # bid = 3 -> bridge = (r0+r1)/2 = midpoint of atop(0,1) and atop(1,1) 
    return out


def hollow_positions(facet, nx, ny, a):
    a1, a2 = lattice_vectors(facet, a)
    cs_all = HOLLOW_CORNERS[_kind(facet)]
    i, j = np.divmod(np.arange(nx * ny), ny)
    out = np.empty((nx * ny * len(cs_all), 2))
    for u, cs in enumerate(cs_all):
        acc = sum((i + di)[:, None] * a1 + (j + dj)[:, None] * a2 for di, dj in cs)
        out[u::len(cs_all)] = acc / len(cs)
    # i.e., facet(111)
    # cs_all = [[(0,0),(1,0),(0,1)], [(1,0),(0,1),(1,1)]] 2 hollow sites per atom
    # list(enumerate(cs_all)) = [ (0, [(0,0),(1,0),(0,1)]), (1, [(1,0),(0,1),(1,1)]) ]
    # hid = 3 -> (i*ny+j)*n_kinds + u -> (i,j,u) = (0,1,1) -> u=1 -> cs = [(1,0),(0,1),(1,1)]
    # for di,dj in cs: (1,0),(0,1),(1,1) -> i=0, j=1 ->atop(0+1,1+0), atop(0+0,1+1), atop(0+1,1+1) -> hollow = centroid of these 3 atop

    return out


def site_masks(facet, nx, ny):
    k = _kind(facet)
    i, j = np.divmod(np.arange(nx * ny), ny)
    # i -> shape (12,); [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # i[:, None] -> shape (12, 1); column vector
    # i[None, :] -> shape (1, 12); row vector
    inside = lambda di, dj: (i + di < nx) & (j + dj < ny)
    bridge_ok = np.stack(
        [inside(*p0) & inside(*p1) for p0, p1 in BRIDGE_OFFSETS[k]], axis=1).ravel() #flattens the 2-D stack into the 1-D array
    hollow_ok = np.stack(
        [np.logical_and.reduce([inside(di, dj) for di, dj in cs])
         for cs in HOLLOW_CORNERS[k]], axis=1).ravel()
    return bridge_ok, hollow_ok


def bridge_atom_ids(facet, nx, ny):
    offs = BRIDGE_OFFSETS[_kind(facet)]
    i, j = np.divmod(np.arange(nx * ny), ny)
    inside = lambda di, dj: (i + di < nx) & (j + dj < ny)
    out = np.full((nx * ny * len(offs), 2), -1, int)
    for d, (p0, p1) in enumerate(offs):
        m = inside(*p0) & inside(*p1)
        for c, (di, dj) in enumerate((p0, p1)):
            out[d::len(offs), c] = np.where(m, (i + di) * ny + (j + dj), -1)
    return out


def n_dirs(facet):  return len(BRIDGE_OFFSETS[_kind(facet)])
def n_kinds(facet): return len(HOLLOW_CORNERS[_kind(facet)])
def atop_id(ny, i, j):             return i * ny + j
def bridge_id(facet, ny, i, j, d): return (i * ny + j) * n_dirs(facet) + d
def hollow_id(facet, ny, i, j, u): return (i * ny + j) * n_kinds(facet) + u

def bridge_ijd(facet, ny, bid):
    cell, d = divmod(bid, n_dirs(facet)); return (*divmod(cell, ny), d)

def hollow_iju(facet, ny, hid):
    cell, u = divmod(hid, n_kinds(facet)); return (*divmod(cell, ny), u)


# ══════════════════════════════════════════════════════════════════════════
#  Lattice
# ══════════════════════════════════════════════════════════════════════════

VACANT, SINGLE_MC, DMC = 0, 1, 2


class Lattice:
    """Carbon on bridges, static H on hollows."""

    def __init__(self, facet="111", nx=5, ny=5, a=2.77, theta_H=0.0,
                 interior_only=False, seed=None):
        self.facet, self.nx, self.ny, self.a = facet, nx, ny, float(a)
        self.rng = np.random.default_rng(seed)

        self.P = atop_positions(facet, nx, ny, a)
        self.B = bridge_positions(facet, nx, ny, a)
        self.H = hollow_positions(facet, nx, ny, a)
        self.BA = bridge_atom_ids(facet, nx, ny)
        self.bridge_ok, self.hollow_ok = site_masks(facet, nx, ny)

        self.d_cc, self.partners, self.n_partners = self._partner_table()

        usable = self.bridge_ok.copy()
        if interior_only:
            usable &= (self.n_partners == self.n_partners.max())
        self.interior_only = bool(interior_only)
        self.usable = usable
        self.sites = np.flatnonzero(usable)          # every usable bridge id
        self.n_sites = int(usable.sum())

        # ── carbon state on bridges ──────────────────────────────────────
        nb = len(self.B)
        self.occupancy = np.zeros(nb, int)           # 0 / 1 / 2
        self.carbon_at_site = np.full(nb, -1, int)
        self.chain_at_site = np.zeros(nb, int)
        self.occupancy[~usable] = -1                 # -1 = site does not exist

        # ── static H on hollows ──────────────────────────────────────────
        self.theta_H = float(theta_H)
        self.hollows = np.flatnonzero(self.hollow_ok)
        self.hollow_H = np.zeros(len(self.hollow_ok), bool)
        n_occ = int(round(self.theta_H * len(self.hollows)))
        if n_occ > 0:
            self.hollow_H[self.rng.choice(self.hollows, n_occ, replace=False)] = True

        self.theta = 0.0
        self.refusals = {"no_free_bridge": 0, "no_free_partner": 0}

    # ── partner table ────────────────────────────────────────────────────

    def _partner_table(self):
        """
        Bridges sharing exactly one metal atom, at the nearest such separation.
        The separation is fixed GLOBALLY (a/2 on hex) rather than per site, so
        a boundary bridge cannot define its own longer "nearest" distance and
        pick up spurious partners.
        """
        nb = len(self.B)
        valid = np.flatnonzero(self.bridge_ok)
        atoms = [set(self.BA[b]) for b in range(nb)]

        share1, dmin = {}, np.inf
        for b in valid:
            lst = []
            for c in valid:
                if c != b and len(atoms[c] & atoms[b]) == 1:
                    d = float(np.linalg.norm(self.B[c] - self.B[b]))
                    lst.append((c, d))
                    dmin = min(dmin, d)
            share1[b] = lst

        partners = np.full((nb, 8), -1, int)
        n_part = np.zeros(nb, int)
        for b, lst in share1.items():
            near = [c for c, d in lst if abs(d - dmin) < 1e-9]
            n_part[b] = len(near)
            partners[b, :len(near)] = near
        return (float(dmin) if np.isfinite(dmin) else 0.0), partners, n_part

    def get_c_neighbors(self, bid):
        """dMC partners of a bridge. reactions.py calls this via self.surface."""
        p = self.partners[int(bid)]
        return p[p >= 0]

    # ── queries: the gate ────────────────────────────────────────────────

    def free_sites(self):
        return np.flatnonzero(self.occupancy == VACANT)

    def has_free_site(self):
        return bool(np.any(self.occupancy == VACANT))

    def free_neighbors(self, bid):
        nb = self.get_c_neighbors(bid)
        return nb[self.occupancy[nb] == VACANT]

    def can_grow_from(self, bid):
        """Gate for dMC: is a partner bridge free next to this carbon?"""
        return bid >= 0 and len(self.free_neighbors(bid)) > 0

    def site_of_carbon(self, carbon_idx):
        hit = np.flatnonzero(self.carbon_at_site == int(carbon_idx))
        return int(hit[0]) if len(hit) else -1

    # ── mutations ────────────────────────────────────────────────────────

    def adsorb(self, carbon_idx, chain_len):
        """Put a carbon on any free bridge. Returns bid, or -1 if refused."""
        cand = self.free_sites()
        if len(cand) == 0:
            self.refusals["no_free_bridge"] += 1
            return -1
        bid = int(self.rng.choice(cand))
        self.occupancy[bid] = SINGLE_MC
        self.carbon_at_site[bid] = int(carbon_idx)
        self.chain_at_site[bid] = int(chain_len)
        self.update_theta()
        return bid

    def grow(self, anchor_bid, carbon_idx, chain_len):
        """
        Second carbon of a dMC pair: must land on a free PARTNER of the anchor.
        Both sites become DMC and carry -N. Returns bid, or -1 if refused.
        """
        cand = self.free_neighbors(anchor_bid)
        if len(cand) == 0:
            self.refusals["no_free_partner"] += 1
            return -1
        bid = int(self.rng.choice(cand))
        self.occupancy[bid] = DMC
        self.occupancy[anchor_bid] = DMC
        self.carbon_at_site[bid] = int(carbon_idx)
        self.chain_at_site[bid] = -int(chain_len)
        self.chain_at_site[anchor_bid] = -int(chain_len)
        self.update_theta()
        return bid

    def release(self, bid):
        """Desorb the carbon at this bridge."""
        bid = int(bid)
        if bid < 0 or self.occupancy[bid] <= VACANT:
            return False
        self.occupancy[bid] = VACANT
        self.carbon_at_site[bid] = -1
        self.chain_at_site[bid] = 0
        self.update_theta()
        return True

    def split(self, bid_left, bid_right, n_left, n_right):
        """
        Scission of the C-C bond between a dMC pair: both carbons stay on the
        surface as single M-C, each now belonging to its own fragment.
        """
        for bid, n in ((int(bid_left), n_left), (int(bid_right), n_right)):
            self.occupancy[bid] = SINGLE_MC
            self.chain_at_site[bid] = int(n)
        self.update_theta()
        return True

    def relabel(self, bid, chain_len):
        """Fragment length at a site changed (e.g. after a scission upstream)."""
        sign = -1 if self.occupancy[int(bid)] == DMC else 1
        self.chain_at_site[int(bid)] = sign * int(chain_len)

    # ── coverage / reporting ─────────────────────────────────────────────

    def update_theta(self):
        occ = int(np.count_nonzero(self.occupancy > VACANT))
        self.theta = occ / self.n_sites if self.n_sites else 0.0
        return self.theta

    @property
    def n_carbons(self):
        return int(np.count_nonzero(self.occupancy > VACANT))

    @property
    def theta_H_actual(self):
        n = len(self.hollows)
        return int(self.hollow_H[self.hollows].sum()) / n if n else 0.0

    def partner_histogram(self):
        v = self.n_partners[self.usable]
        return {int(k): int((v == k).sum()) for k in sorted(set(v.tolist()))}

    def summary(self):
        return (f"{self.facet} {self.nx}x{self.ny}  a={self.a:.2f} A\n"
                f"  bridges {len(self.B)} ids -> {int(self.bridge_ok.sum())} valid"
                f" -> {self.n_sites} usable"
                f"{'  (interior only)' if self.interior_only else ''}\n"
                f"  hollows {len(self.H)} ids -> {len(self.hollows)} valid\n"
                f"  d_CC = {self.d_cc:.4f} A   dMC partners per usable bridge: "
                f"{self.partner_histogram()}\n"
                f"  C {self.n_carbons}  theta_C = {self.theta:.4f}   "
                f"H {int(self.hollow_H.sum())}  theta_H = {self.theta_H:.4f} "
                f"(realised {self.theta_H_actual:.4f})\n"
                f"  refusals {self.refusals}")

    # ── plot ─────────────────────────────────────────────────────────────

    def plot(self, ax=None, show_ids=False, r_frac=0.5, bonds=True):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 6.5))
        for k, (x, y) in enumerate(self.P):
            ax.add_patch(Circle((x, y), r_frac * self.a, facecolor="0.82",
                                edgecolor="0.45", lw=1.0, zorder=1))
            if show_ids:
                ax.text(x, y, str(k), ha="center", va="center", fontsize=7,
                        color="0.35", zorder=3)

        free = np.flatnonzero(self.occupancy == VACANT)
        ax.plot(self.B[free, 0], self.B[free, 1], "s", color="black", ms=4,
                zorder=4, label=f"free bridge ({len(free)})")

        hf = self.hollows[~self.hollow_H[self.hollows]]
        ho = self.hollows[self.hollow_H[self.hollows]]
        ax.plot(self.H[hf, 0], self.H[hf, 1], "o", color="red", ms=4, zorder=4,
                label=f"empty hollow ({len(hf)})")
        if len(ho):
            ax.plot(self.H[ho, 0], self.H[ho, 1], "o", color="tab:green", ms=8,
                    mec="k", zorder=5, label=f"H on hollow ({len(ho)})")

        occ = np.flatnonzero(self.occupancy > VACANT)
        if bonds:
            for m, b1 in enumerate(occ):
                for b2 in occ[m + 1:]:
                    if abs(self.carbon_at_site[b1] - self.carbon_at_site[b2]) == 1:
                        ax.plot(*zip(self.B[b1], self.B[b2]), "-",
                                color="tab:blue", lw=3, zorder=6)
        if len(occ):
            dm = occ[self.occupancy[occ] == DMC]
            sg = occ[self.occupancy[occ] == SINGLE_MC]
            if len(sg):
                ax.plot(self.B[sg, 0], self.B[sg, 1], "o", color="tab:blue",
                        ms=14, mec="k", zorder=7, label=f"single M-C ({len(sg)})")
            if len(dm):
                ax.plot(self.B[dm, 0], self.B[dm, 1], "o", color="tab:purple",
                        ms=14, mec="k", zorder=7, label=f"dMC ({len(dm)})")
            for b in occ:
                x, y = self.B[b]
                ax.text(x, y, str(self.carbon_at_site[b]), ha="center",
                        va="center", fontsize=7, color="white",
                        fontweight="bold", zorder=8)

        ax.set_aspect("equal"); ax.autoscale_view(); ax.margins(0.15)
        ax.set_xlabel("x  [Å]"); ax.set_ylabel("y  [Å]")
        ax.set_title(f"{self.facet}  {self.nx}×{self.ny}   "
                     f"θ_C = {self.theta:.3f}   θ_H = {self.theta_H:.3f}")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        return ax


if __name__ == "__main__":
    lat = Lattice("111", 5, 5, 2.77, theta_H=0.5794, seed=0)
    print(lat.summary())

    b = lat.adsorb(1, 4)
    print(f"\nadsorb carbon 1 of a C4 -> bridge {b}, "
          f"free partners {len(lat.free_neighbors(b))}")
    b2 = lat.grow(b, 2, 4)
    shared = set(lat.BA[b]) & set(lat.BA[b2])
    print(f"dMC: carbon 2 -> bridge {b2}; share {len(shared)} metal atom, "
          f"d = {np.linalg.norm(lat.B[b] - lat.B[b2]):.4f} A, "
          f"occupancy {lat.occupancy[b]}/{lat.occupancy[b2]}, "
          f"chain_at_site {lat.chain_at_site[b]}/{lat.chain_at_site[b2]}")
    lat.split(b, b2, 2, 2)
    print(f"crack -> occupancy {lat.occupancy[b]}/{lat.occupancy[b2]}, "
          f"chain_at_site {lat.chain_at_site[b]}/{lat.chain_at_site[b2]}")
    print(f"theta_C = {lat.theta:.4f}")