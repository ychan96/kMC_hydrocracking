import numpy as np

VACANT, SINGLE_MC, DOUBLE_MC = 0, 1, 2


class CoverageMixin:
    def metal_surface(self, reaction_key, chain_info):
        ok = {'adsorption': self.coverage_adsorption,
              'desorption': self.coverage_desorption,
              'dmc':        self.coverage_dmc_formation,
              'cracking':   self.coverage_cracking}[reaction_key](chain_info)
        self.update_theta()
        return ok

    def update_theta(self):
        self.theta = np.count_nonzero(self.m_bond) / self.m_bond.size
        return self.theta

    def has_vacant_site(self):
        return bool(np.any(self.m_bond == VACANT))

    def _neighbors(self, i, j):
        n = self.m_size
        return [(r, c) for r in range(max(0, i - 1), min(n, i + 2))
                for c in range(max(0, j - 1), min(n, j + 2)) if (r, c) != (i, j)]

    def _sites(self, bond, chain_value):
        return [tuple(map(int, s)) for s in np.argwhere((self.m_bond == bond) & (self.m_chain == chain_value))]

    def _dmc_anchors(self, N):
        out = []
        for i, j in self._sites(SINGLE_MC, N):
            vac = [s for s in self._neighbors(i, j) if self.m_bond[s] == VACANT]
            if vac:
                out.append(((i, j), vac))
        return out

    def can_form_dmc(self, N):
        return len(self._dmc_anchors(N)) > 0

    def _pick(self, seq):
        return seq[np.random.randint(len(seq))]

    def coverage_adsorption(self, N):
        vac = np.argwhere(self.m_bond == VACANT)
        if len(vac) == 0:
            return False
        i, j = vac[np.random.randint(len(vac))]
        self.m_bond[i, j], self.m_chain[i, j] = SINGLE_MC, N
        return True

    def coverage_desorption(self, N):
        sites = self._sites(SINGLE_MC, N)
        if not sites:
            return False
        s = self._pick(sites)
        self.m_bond[s], self.m_chain[s] = VACANT, 0
        return True

    def coverage_dmc_formation(self, N):
        anchors = self._dmc_anchors(N)
        if not anchors:
            return False
        a, vac = self._pick(anchors)
        b = self._pick(vac)
        for s in (a, b):
            self.m_bond[s], self.m_chain[s] = DOUBLE_MC, -N
        self.dmc_pairs.append((a, b, N))
        return True

    def coverage_cracking(self, info):
        N, n1, n2 = info
        idx = [k for k, p in enumerate(self.dmc_pairs) if p[2] == N]
        if not idx:
            return False
        a, b, _ = self.dmc_pairs.pop(self._pick(idx))
        self.m_bond[a], self.m_chain[a] = SINGLE_MC, n1
        self.m_bond[b], self.m_chain[b] = SINGLE_MC, n2
        return True