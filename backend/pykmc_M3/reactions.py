import numpy as np

from .constants import cn
from .lattice import SMC, DMC


class ReactionMixin:
    def select_reaction(self, counts):
        th    = float(self.theta_H)
        av_v = (1.0 - th) ** 2          # ads, dmc
        av_h = th ** 2                  # des, crk

        rates = {}
        for key, n in counts.items():
            if n <= 0:
                continue
            w = av_v if key[:3] in ('ads', 'dmc') else av_h
            r = self.k_const[key] * n * w
            if r > 0.0:
                rates[key] = r

        R = sum(rates.values())
        if R <= 0.0:
            return None, 0.0

        keys = list(rates.keys())
        cdf  = np.cumsum([rates[k] for k in keys]) / R

        u1, u2 = np.random.rand(), np.random.rand()

        idx = int(np.searchsorted(cdf, u1, side='right'))
        idx = min(idx, len(keys) - 1)          # u1 == 1.0 guard

        dt = -np.log(u2) / R
        return keys[idx], dt

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def perform_reaction(self, reaction_key):
        rxn, _, tag = reaction_key.partition('_')

        if rxn == 'ads':
            return self.perform_adsorption(tag)
        if rxn == 'des':
            return self.perform_desorption(tag)
        if rxn == 'dmc':
            return self.perform_dmc_formation(tag)
        if rxn == 'crk':
            return self.perform_cracking(tag)
        return False

    # ------------------------------------------------------------------
    # Adsorption
    # ------------------------------------------------------------------

    def perform_adsorption(self, tag):
        pool = []
        for start, end in self.chains:
            L = end - start
            if L == 1 or cn(L) != tag:
                continue
            if np.any(self.carbon_array[start:end] == 1):
                continue
            pool.extend((c, L) for c in range(start, end))

        if not pool:
            return False

        global_c, chain_len = pool[np.random.randint(len(pool))]

        bid = self.surface.adsorb(global_c, chain_len)
        if bid < 0:
            return False

        self.carbon_array[global_c]   = 1
        self.carbon_to_site[global_c] = bid
        return True

    # ------------------------------------------------------------------
    # Desorption
    # ------------------------------------------------------------------

    def perform_desorption(self, tag):
        cands = []
        for start, end in self.chains:
            L = end - start
            if cn(L) != tag:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1:
                continue
            global_c = start + int(np.where(seg == 1)[0][0])
            bid      = int(self.carbon_to_site[global_c])
            if bid < 0 or self.occupancy[bid] != SMC:
                continue
            cands.append((global_c, bid))

        if not cands:
            return False

        global_c, bid = cands[np.random.randint(len(cands))]

        if not self.surface.release(bid):
            return False

        self.carbon_array[global_c]   = 0
        self.carbon_to_site[global_c] = -1
        return True

    # ------------------------------------------------------------------
    # dMC formation
    # ------------------------------------------------------------------

    def perform_dmc_formation(self, tag):
        want_t = (tag == 't')
        cands  = []

        for start, end in self.chains:
            L = end - start
            if L == 1:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1:
                continue

            a      = int(np.where(seg == 1)[0][0])
            anchor = int(self.carbon_to_site[start + a])
            if anchor < 0 or not self.surface.can_grow_from(anchor):
                continue

            left_vacant  = a > 0 and seg[a - 1] == 0
            right_vacant = a < L - 1 and seg[a + 1] == 0

            if L in (2, 3):
                if left_vacant and want_t:
                    cands.append((start + a - 1, anchor, L))
                if right_vacant and want_t:
                    cands.append((start + a + 1, anchor, L))
                continue

            if left_vacant:
                bond_pos = a - 1
                is_t = (bond_pos == 0) or (a == L - 1)
                if is_t == want_t:
                    cands.append((start + a - 1, anchor, L))

            if right_vacant:
                bond_pos = a
                is_t = (bond_pos == 0) or (bond_pos == L - 2)
                if is_t == want_t:
                    cands.append((start + a + 1, anchor, L))

        if not cands:
            return False

        global_nb, anchor, L = cands[np.random.randint(len(cands))]

        bid = self.surface.grow(anchor, global_nb, L)
        if bid < 0:
            return False

        self.carbon_array[global_nb]   = 1
        self.carbon_to_site[global_nb] = bid
        return True

    # ------------------------------------------------------------------
    # Cracking
    # ------------------------------------------------------------------

    def perform_cracking(self, tag):
        want_t = (tag == 't')
        cands  = []

        for start, end in self.chains:
            L = end - start
            if L == 1:
                continue
            seg = self.carbon_array[start:end]

            for i in range(L - 1):
                if seg[i] != 1 or seg[i + 1] != 1:
                    continue
                bl = int(self.carbon_to_site[start + i])
                br = int(self.carbon_to_site[start + i + 1])
                if bl < 0 or br < 0:
                    continue
                if self.occupancy[bl] != DMC or self.occupancy[br] != DMC:
                    continue
                is_t = (i == 0) or (i == L - 2)
                if is_t == want_t:
                    cands.append((start, end, i, bl, br))

        if not cands:
            return False

        start, end, i, bl, br = cands[np.random.randint(len(cands))]

        chain_idx = start + i + 1            # chain_array index of the bond
        n_left    = chain_idx - start
        n_right   = end - chain_idx

        self.surface.split(bl, br, n_left, n_right)
        self.chain_array[chain_idx] = 0

        self.invalidate_chains()
        return True