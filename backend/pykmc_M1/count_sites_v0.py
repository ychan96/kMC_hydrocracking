import numpy as np


class ConfigMixin:
    def _identify_chains(self):
        breaks = np.flatnonzero(self.chain_array == 0)
        return [(int(a), int(b)) for a, b in zip(breaks[:-1], breaks[1:]) if b > a]

    def update_configuration(self):
        counts = {'adsorption': 0, 'desorption': 0, 'dmc': 0, 'cracking': 0}
        has_vacant = self.has_vacant_site()

        for start, end in self.chains:
            N   = end - start
            seg = self.carbon_array[start:end]
            n_ads = int(seg.sum())

            if n_ads == 0:
                if N > 1 and has_vacant:
                    counts['adsorption'] += N
            elif n_ads == 1:
                counts['desorption'] += 1
                if N > 1 and self.can_form_dmc(N):
                    i = int(np.flatnonzero(seg)[0])
                    counts['dmc'] += int(i > 0 and seg[i - 1] == 0) + int(i < N - 1 and seg[i + 1] == 0)
            elif n_ads == 2:
                counts['cracking'] += int(np.sum(seg[:-1] & seg[1:]))

        return counts