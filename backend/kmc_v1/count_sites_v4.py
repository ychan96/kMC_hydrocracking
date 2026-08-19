import numpy as np
from kmc_v1.init_v4 import BaseKineticMC


class ConfigMixin:
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
        ads_keys = {'ads_c2': 0}
        for n in range(3, 31):
            ads_keys[f'ads_c{n}'] = 0
        ads_keys['ads_c31plus'] = 0

        # ── desorption keys ───────────────────────────────────────
        des_keys = {'des_c1': 0, 'des_c2': 0}
        for n in range(3, 31):
            des_keys[f'des_c{n}'] = 0
        des_keys['des_c31plus'] = 0

        # ── dMC & cracking (length-integrated) ───────────────────
        rxn_keys = {'dmc_t': 0, 'dmc_i': 0, 'crk_t': 0, 'crk_i': 0}

        counts = {**ads_keys, **des_keys, **rxn_keys}

        for start, end in self.chains:
            chain_length  = end - start
            chain_segment = self.carbon_array[start:end]

            self._count_adsorption_sites(chain_segment, chain_length, counts)
            self._count_desorption_sites(chain_segment, chain_length, counts)
            self._count_dmc_sites(chain_segment, chain_length, counts)
            self._count_cracking_sites(chain_segment, chain_length, counts)

        return counts

    # ── chain identification ──────────────────────────────────────

    def _identify_chains(self):
        """
        Identify separate chains by checking connectivity in chain_array.
        Returns list of (start, end) tuples for each chain.
        """
        chains = []
        current_chain = [0]

        for i in range(1, len(self.chain_array)):
            if self.chain_array[i] == 1:
                current_chain.append(i)
            else:
                chains.append((current_chain[0], current_chain[-1] + 1))
                current_chain = [i]

        return chains

    # ── key routing helper ────────────────────────────────────────

    @staticmethod
    def _cn(n):
        """Return key infix for chain length n (capped at c31plus)."""
        return f'c{n}' if n <= 30 else 'c31plus'

    # ── adsorption ────────────────────────────────────────────────

    def _count_adsorption_sites(self, chain_segment, chain_length, counts):
        """
        C1  : excluded
        C2  : terminal only (both ends of a 2-carbon chain)
        C3+ : internal  → indices [1:-1]
               terminal → indices [0] and [-1]
        Only counts if chain is entirely unadsorbed.
        """
        if chain_length == 1:
            return

        if np.any(chain_segment == 1):
            return

        cn = self._cn(chain_length)
        counts[f'ads_{cn}'] += chain_length

    # ── desorption ────────────────────────────────────────────────

    def _count_desorption_sites(self, chain_segment, chain_length, counts):
        """
        Requires exactly one carbon adsorbed.
        C1, C2 : terminal desorption only
        C3+    : terminal or internal depending on adsorbed position
        """
        if np.sum(chain_segment == 1) != 1:
            return

        counts[f'des_{self._cn(chain_length)}'] += 1

    # ── dMC ───────────────────────────────────────────────────────

    def _count_dmc_sites(self, chain_segment, chain_length, counts):
        """
        Requires exactly one carbon adsorbed; counts vacant neighbours.

        C1      : excluded
        C2, C3  : all bonds are terminal  → dmc_t
        C4+     : classify each potential bond:
                    bond touches chain end → dmc_t
                    bond fully internal    → dmc_i
        """
        if chain_length == 1:
            return

        if np.sum(chain_segment == 1) != 1:
            return

        attached_idx = np.where(chain_segment == 1)[0][0]
        left_vacant  = attached_idx > 0 and chain_segment[attached_idx - 1] == 0
        right_vacant = attached_idx < chain_length - 1 and chain_segment[attached_idx + 1] == 0

        if not left_vacant and not right_vacant:
            return

        if chain_length in (2, 3):
            counts['dmc_t'] += int(left_vacant) + int(right_vacant)
            return

        # C4+
        if left_vacant:
            bond_pos = attached_idx - 1
            if bond_pos == 0 or attached_idx == chain_length - 1:
                counts['dmc_t'] += 1
            else:
                counts['dmc_i'] += 1

        if right_vacant:
            bond_pos = attached_idx
            if bond_pos == 0 or bond_pos == chain_length - 2:
                counts['dmc_t'] += 1
            else:
                counts['dmc_i'] += 1

    # ── cracking ──────────────────────────────────────────────────

    def _count_cracking_sites(self, chain_segment, chain_length, counts):
        """
        Find all '11' patterns (two adjacent adsorbed carbons).

        C1  : excluded
        C2+ : bond at index 0 or (chain_length-2) → crk_t
              otherwise                           → crk_i
        """
        if chain_length == 1:
            return

        for i in range(chain_length - 1):
            if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                is_terminal = (i == 0) or (i == chain_length - 2)
                if is_terminal:
                    counts['crk_t'] += 1
                else:
                    counts['crk_i'] += 1