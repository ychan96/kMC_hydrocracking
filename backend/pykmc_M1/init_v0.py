import numpy as np


class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, chain_length=None,
                 rate_constants=None, P_H2=50, m_size=5):
        self.temp_K        = temp_C + 273.15
        self.reaction_time = reaction_time
        self.P_H2          = P_H2
        self.K_H           = 1.43
        self.m_size        = m_size
        self.k_const       = rate_constants or {'ads': 3.5163410884574137e-04,
                                                'des': 8.464077690465607e-03,
                                                'dmc': 9.84312312312323e-04,
                                                'crk': 8.984746813456814e-03}

        self.m_bond  = np.zeros((m_size, m_size), int)
        self.m_chain = np.zeros((m_size, m_size), int)
        self.theta   = 0.0
        self.dmc_pairs = []

        if chain_length is None:
            chain_length = int(np.floor(np.random.normal(280, 10)))
        self.chain_length   = int(chain_length)
        self.carbon_array   = np.zeros(self.chain_length, int)
        self.chain_array    = np.zeros(self.chain_length + 1, int)
        self.chain_array[1:-1] = 1

        self.current_time  = 0.0
        self._chains       = None
        self._chains_valid = False

    @property
    def chains(self):
        if not self._chains_valid:
            self._chains       = self._identify_chains()
            self._chains_valid = True
        return self._chains

    def invalidate_chains(self):
        self._chains_valid = False

    @property
    def theta_H(self):
        x = np.sqrt(self.K_H * self.P_H2)
        return x / (1.0 + x)