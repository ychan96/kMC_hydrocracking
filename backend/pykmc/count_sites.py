import numpy as np
from collections import defaultdict
from .constants import MAX_LENGTH, C_LUMP, cn


class ConfigMixin:
    """
    Mixin for chain identification and reaction-site counting.

    Counting output (from update_configuration) is keyed by current fragment
    length N so that reactions.py can compute one Arrhenius rate per (type, N)
    and multiply by the site count.

    Relies on BaseKineticMC arrays:
        carbon_array[j]   : 0=free, 1=adsorbed
        chain_array[i]    : 0=break/boundary, 1=bonded  (length N_total + 1)
        occupancy[i]      : 0=vacant, 1=single M-C, 2=dMC
        carbon_to_site[j] : surface site index for carbon j (-1 if free)
        self.surface      : CatalystSurface (provides get_c_neighbors)
    """     
    # ------------------------------------------------------------------
    # Chain identification
    # ------------------------------------------------------------------

    def _identify_chains(self):
        """
        Scan chain_array for breaks (value 0) to produce contiguous fragments.

        chain_array has length N_total + 1.
        chain_array[i] == 1 means carbon[i-1] and carbon[i] are bonded.
        Boundaries (index 0 and N_total) are always 0.

        Returns:
            List of (start, end) tuples — carbon_array[start:end] is one fragment.
        """
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
        """
        Count available sites for each reaction type across all chain lengths.
        Returns a dictionary of counts organized by reaction type and chain length.

        Key scheme
        ----------
        ads : C1 excluded; C2+ single key per chain length (ads_cN);
              C31+  catch-all (ads_c31plus)
        des : C1+ single key per chain length (des_cN);
              C31+  catch-all (des_c31plus)
        dmc : length-integrated  →  dmc
        crk : length-integrated  →  crk
        """
        # ── adsorption keys ───────────────────────────────────────
        ads_keys = {f'ads_{cn(n)}': 0 for n in range(2, MAX_LENGTH + 1)}
        ads_keys[f'ads_{cn(C_LUMP)}'] = 0

        # ── desorption keys ───────────────────────────────────────
        des_keys = {f'des_{cn(n)}': 0 for n in range(1, MAX_LENGTH + 1)}
        des_keys[f'des_{cn(C_LUMP)}'] = 0

        # ── dMC & cracking (length-integrated) ───────────────────
        rxn_keys = {'dmc_t': 0, 'dmc_i': 0, 'crk_t': 0, 'crk_i': 0}

        counts = {**ads_keys, **des_keys, **rxn_keys}

        for start, end in self.chains:
            chain_len  = end - start
            chain_segment = self.carbon_array[start:end]

            self._count_adsorption_sites(chain_segment, chain_len, counts)
            self._count_desorption_sites(chain_segment, chain_len, counts)
            self._count_dmc_sites(chain_segment, chain_len, counts)
            self._count_cracking_sites(chain_segment, chain_len, counts)

        return counts

    # ── adsorption ────────────────────────────────────────────────

    def _count_adsorption_sites(self, chain_segment, chain_len, counts):
        """
        C1  : excluded
        C2+ : single key per chain length (ads_cN)
        C31+: catch-all (ads_c31plus)

        Only counts if chain is entirely unadsorbed.
        """
        if chain_len == 1:
            return

        if np.any(chain_segment == 1):
            return
        num = cn(chain_len)
        counts[f'ads_{num}'] += chain_len

    # ── desorption ────────────────────────────────────────────────

    def _count_desorption_sites(self, chain_segment, chain_len, counts):
        """
        Requires exactly one carbon adsorbed.
        C1, C2 : terminal desorption only
        C3+    : terminal or internal depending on adsorbed position
        """
        if np.sum(chain_segment == 1) != 1:
            return

        counts[f'des_{self._cn(chain_len)}'] += 1

    # ── dMC ───────────────────────────────────────────────────────

    def _count_dmc_sites(self, chain_segment, chain_len, counts):
        """
        Requires exactly one carbon adsorbed; counts vacant neighbours.

        C1      : excluded
        C2, C3  : all bonds are terminal  → dmc_t
        C4+     : classify each potential bond:
                    bond touches chain end → dmc_t
                    bond fully internal    → dmc_i
        """
        if chain_len == 1:
            return

        if np.sum(chain_segment == 1) != 1:
            return

        attached_idx = np.where(chain_segment == 1)[0][0]
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
        """
        Find all '11' patterns (two adjacent adsorbed carbons).

        C1  : excluded
        C2+ : bond at index 0 or (chain_len-2) → crk_t
              otherwise                        → crk_i
        """
        if chain_len == 1:
            return

        for i in range(chain_len - 1):
            if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                is_terminal = (i == 0) or (i == chain_len - 2)
                if is_terminal:
                    counts['crk_t'] += 1
                else:
                    counts['crk_i'] += 1