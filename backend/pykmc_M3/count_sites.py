import numpy as np
from .constants import MAX_LENGTH, C_LUMP, cn
from .lattice import SMC


class ConfigMixin:
    def _identify_chains(self):
        chains      = []
        chain_start = 0
        n_carbons   = len(self.carbon_array)

        for i in range(1, n_carbons + 1):
            if self.chain_array[i] == 0:
                if i > chain_start:
                    chains.append((chain_start, i))
                chain_start = i

        return chains

    # ------------------------------------------------------------------
    # Top-level counter
    # ------------------------------------------------------------------

    def update_configuration(self):
        # ── adsorption keys ───────────────────────────────────────
        ads_keys = {f'ads_{cn(n)}': 0 for n in range(2, MAX_LENGTH + 1)}
        ads_keys[f'ads_{cn(C_LUMP)}'] = 0

        # ── desorption keys ───────────────────────────────────────
        des_keys = {f'des_{cn(n)}': 0 for n in range(1, MAX_LENGTH + 1)}
        des_keys[f'des_{cn(C_LUMP)}'] = 0

        # ── dMC & cracking (length-integrated) ───────────────────
        rxn_keys = {'dmc_t': 0, 'dmc_i': 0, 'crk_t': 0, 'crk_i': 0}

        counts = {**ads_keys, **des_keys, **rxn_keys}

        # lattice gate for adsorption: evaluated once per sweep
        free_bridge = self.surface.has_free_site()

        for start, end in self.chains:
            chain_len     = end - start
            chain_segment = self.carbon_array[start:end]

            if free_bridge:
                self._count_adsorption_sites(chain_segment, chain_len, counts)
            self._count_desorption_sites(chain_segment, chain_len, counts, start)
            self._count_dmc_sites(chain_segment, chain_len, counts, start)
            self._count_cracking_sites(chain_segment, chain_len, counts)

        return counts

    # ── adsorption ────────────────────────────────────────────────

    def _count_adsorption_sites(self, chain_segment, chain_len, counts):
        if chain_len == 1:
            return

        if np.any(chain_segment == 1):
            return
        num = cn(chain_len)
        counts[f'ads_{num}'] += chain_len

    # ── desorption ────────────────────────────────────────────────

    def _count_desorption_sites(self, chain_segment, chain_len, counts, start):
        if np.sum(chain_segment == 1) != 1:
            return

        attached_idx = int(np.where(chain_segment == 1)[0][0])
        bid = int(self.carbon_to_site[start + attached_idx])
        if bid < 0 or self.occupancy[bid] != SMC:
            return

        counts[f'des_{cn(chain_len)}'] += 1

    # ── dMC ───────────────────────────────────────────────────────

    def _count_dmc_sites(self, chain_segment, chain_len, counts, start):
        if chain_len == 1:
            return

        if np.sum(chain_segment == 1) != 1:
            return

        attached_idx = int(np.where(chain_segment == 1)[0][0])

        # lattice gate
        anchor_bid = int(self.carbon_to_site[start + attached_idx])
        if not self.surface.can_grow_from(anchor_bid):
            return

        left_vacant  = attached_idx > 0 and chain_segment[attached_idx - 1] == 0
        right_vacant = attached_idx < chain_len - 1 and chain_segment[attached_idx + 1] == 0

        if not left_vacant and not right_vacant:
            return

        if chain_len in (2, 3):
            counts['dmc_t'] += int(left_vacant) + int(right_vacant)
            return

        # C4+
        if left_vacant:
            bond_pos = attached_idx - 1
            if bond_pos == 0 or attached_idx == chain_len - 1:
                counts['dmc_t'] += 1
            else:
                counts['dmc_i'] += 1

        if right_vacant:
            bond_pos = attached_idx
            if bond_pos == 0 or bond_pos == chain_len - 2:
                counts['dmc_t'] += 1
            else:
                counts['dmc_i'] += 1

    # ── cracking ──────────────────────────────────────────────────

    def _count_cracking_sites(self, chain_segment, chain_len, counts):
        if chain_len == 1:
            return

        for i in range(chain_len - 1):
            if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                is_terminal = (i == 0) or (i == chain_len - 2)
                if is_terminal:
                    counts['crk_t'] += 1
                else:
                    counts['crk_i'] += 1