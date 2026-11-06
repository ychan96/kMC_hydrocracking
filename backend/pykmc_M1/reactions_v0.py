import numpy as np


class ReactionMixin:
    def select_reaction(self, counts):
        k_key = {'adsorption': 'ads', 'desorption': 'des', 'dmc': 'dmc', 'cracking': 'crk'}
        keys  = [k for k, n in counts.items() if n > 0]
        rates = np.array([self.k_const[k_key[k]] * counts[k] for k in keys], float)
        R = rates.sum()
        if R <= 0:
            return None, 0.0

        idx = min(int(np.searchsorted(np.cumsum(rates) / R, np.random.rand(), side='right')), len(keys) - 1)
        return keys[idx], -np.log(np.random.rand()) / R

    def perform_reaction(self, reaction_key):
        return {'adsorption': self.perform_adsorption,
                'desorption': self.perform_desorption,
                'dmc':        self.perform_dmc_formation,
                'cracking':   self.perform_cracking}[reaction_key]()

    @staticmethod
    def _pick(candidates):
        return candidates[np.random.randint(len(candidates))]

    def perform_adsorption(self):
        cands = [(c, end - start) for start, end in self.chains
                 if end - start > 1 and not self.carbon_array[start:end].any()
                 for c in range(start, end)]
        if not cands:
            return False, None
        c, N = self._pick(cands)
        self.carbon_array[c] = 1
        return True, N

    def perform_desorption(self):
        cands = []
        for start, end in self.chains:
            seg = self.carbon_array[start:end]
            if seg.sum() == 1:
                cands.append((start + int(np.flatnonzero(seg)[0]), end - start))
        if not cands:
            return False, None
        c, N = self._pick(cands)
        self.carbon_array[c] = 0
        return True, N

    def perform_dmc_formation(self):
        cands = []
        for start, end in self.chains:
            N   = end - start
            seg = self.carbon_array[start:end]
            if N == 1 or seg.sum() != 1 or not self.can_form_dmc(N):
                continue
            i = int(np.flatnonzero(seg)[0])
            cands += [(start + nb, N) for nb in (i - 1, i + 1) if 0 <= nb < N and seg[nb] == 0]
        if not cands:
            return False, None
        c, N = self._pick(cands)
        self.carbon_array[c] = 1
        return True, N

    def perform_cracking(self):
        cands = []
        for start, end in self.chains:
            seg = self.carbon_array[start:end]
            if seg.sum() != 2:
                continue
            cands += [(start, end, i) for i in range(end - start - 1) if seg[i] and seg[i + 1]]
        if not cands:
            return False, None
        start, end, i = self._pick(cands)
        self.chain_array[start + i + 1] = 0
        self.invalidate_chains()
        return True, (end - start, i + 1, end - start - i - 1)