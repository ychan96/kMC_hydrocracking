import numpy as np
from scipy import constants

eV = 1.60218e-19

class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, chain_length=None, 
                 rate_constants=None, P_H2=50, m_size=5): #desorption_penalty=0.0):
        # Parameters
        self.temp_K = temp_C + 273.15
        self.kb_eV = constants.Boltzmann / constants.e
        self.reaction_time = reaction_time
        self.P_H2 = P_H2  # FIX 1: Store P_H2 (you're accepting it but not storing it)
        
        # FIX 2: Initialize m_size and active sites BEFORE init_arrays
        # because init_arrays might need to know the surface geometry
        self.m_size = m_size
        #self.b = desorption_penalty
        self.init_active_sites(m_size)
        
        # Initialize arrays
        if chain_length is None:
            chain_length = self.normal_dist(mu=280, sigma=10, n_samples=None)
        
        self.chain_length = chain_length
        self.init_arrays(chain_length)
        
        # Rate constant system:
        #   C1, C2        : terminal ads/des only (2 params each)
        #   C3 ~ C30      : internal + terminal ads/des (4 params each)
        #   dMC, cracking : internal + terminal per chain length C3~C30 (2 params each)
        #   C31+          : internal + terminal ads/des (4 params, catch-all)
        if rate_constants is not None:
            self.k_const = rate_constants
        else:
            self.k_const = {
                'ads_c1': 2.707519e-04,
                'des_c1': 2.124751e-02,
                'ads_c2': 4.812345e-04,
                'des_c2': 9.155617e-04,
                'ads_c3': 7.332838e-04,
                'des_c3': 3.948630e-05,
                'ads_c4': 1.239785e-04,
                'des_c4': 1.399816e-03,
                'ads_c5': 6.040204e-05,
                'des_c5': 1.864751e-03,
                'ads_c6': 2.920404e-05,
                'des_c6': 2.485881e-03,
                'ads_c7': 1.407999e-05,
                'des_c7': 3.315966e-03,
                'ads_c8': 6.781263e-06,
                'des_c8': 4.426089e-03,
                'ads_c9': 3.264720e-06,
                'des_c9': 5.911165e-03,
                'ads_c10': 1.571537e-06,
                'des_c10': 7.899454e-03,
                'ads_c11': 7.564557e-07,
                'des_c11': 1.056219e-02,
                'ads_c12': 1.396801e-04,
                'des_c12': 5.749607e-02,
                'ads_c13': 1.686135e-04,
                'des_c13': 5.236975e-02,
                'ads_c14': 2.035359e-04,
                'des_c14': 4.772293e-02,
                'ads_c15': 2.456933e-04,
                'des_c15': 4.350764e-02,
                'ads_c16': 2.965830e-04,
                'des_c16': 3.968173e-02,
                'ads_c17': 3.580112e-04,
                'des_c17': 3.620644e-02,
                'ads_c18': 4.321646e-04,
                'des_c18': 3.304803e-02,
                'ads_c19': 5.216607e-04,
                'des_c19': 3.017600e-02,
                'ads_c20': 6.297113e-04,
                'des_c20': 2.756337e-02,
                'ads_c21': 7.601646e-04,
                'des_c21': 2.518489e-02,
                'ads_c22': 9.175925e-04,
                'des_c22': 2.301894e-02,
                'ads_c23': 1.107672e-03,
                'des_c23': 2.104578e-02,
                'ads_c24': 1.337070e-03,
                'des_c24': 1.924655e-02,
                'ads_c25': 1.614016e-03,
                'des_c25': 1.760646e-02,
                'ads_c26': 1.948341e-03,
                'des_c26': 1.610989e-02,
                'ads_c27': 2.351855e-03,
                'des_c27': 1.474418e-02,
                'ads_c28': 2.838997e-03,
                'des_c28': 1.349756e-02,
                'ads_c29': 3.427027e-03,
                'des_c29': 1.235872e-02,
                'ads_c30': 4.136862e-03,
                'des_c30': 1.131853e-02,
                'ads_c31plus': 4.993724e-03,
                'des_c31plus': 1.036820e-02,
		        'dmc_i': 2.061422e-02,
                'dmc_t': 2.18325e-03,
                'crk_i': 9.641641e-04,
                'crk_t': 1.946220e-03,
            }
    

        
        self.current_time = 0.0
        self._chains = None
        self._chains_valid = False
    
    @property  # FIX 3: Move @property decorator before the method definition
    def chains(self):
        """Lazy evaluation of chain identification"""
        if not self._chains_valid or self._chains is None: #when self._chains_valid is False -> clear cache
            self._chains = self._identify_chains()
            self._chains_valid = True
        return self._chains
    
    # ── Rate lookup helpers ───────────────────────────────────────

    def _cn(self, n):
        """Return the key suffix for chain length n (capped at c31plus)."""
        if n <= 30:
            return f'c{n}'
        return 'c31plus'

    def get_adsorption_rate(self, chain_length, is_terminal=None):
        return self.k_const[f'ads_{self._cn(chain_length)}']

    def get_desorption_rate(self, chain_length, is_terminal=None):
        return self.k_const[f'des_{self._cn(chain_length)}']

    def get_dmc_rate(self, chain_length, is_terminal):
        """Returns None for C1 (no dMC reaction)."""
        if chain_length == 1:
            return None
        return self.k_const['dmc_t'] if is_terminal else self.k_const['dmc_i']

    def get_cracking_rate(self, chain_length, is_terminal):
        """Returns None for C1 (no C-C bonds)."""
        if chain_length == 1:
            return None
        return self.k_const['crk_t'] if is_terminal else self.k_const['crk_i']
    
    def normal_dist(self, mu=260, sigma=30, n_samples=None):
        """Draw a sample chain length from a normal distribution"""
        size = n_samples or 1 
        x = np.random.normal(loc=mu, scale=sigma, size=size)
        chain = int(np.floor(x[0]))
        return chain

    def init_arrays(self, chain_length):
        """Initialize carbon and chain tracking arrays"""
        self.carbon_array = np.zeros(chain_length, int)
        self.chain_array = np.zeros(chain_length + 1, int)
        self.chain_array[1:-1] = 1
        # -1 means not bound, (i,j) tuple means bound at that site
        self.carbon_to_site = np.full(chain_length, -1, dtype=object)
        # Track which carbon is at each site (inverse mapping)
        self.site_to_carbon = {}  # {(row, col): carbon_index}
        
    def init_active_sites(self, m_size):
        n = m_size
        total_sites = n * n
        
        # Two separate matrices
        self.m_bond  = np.zeros((n, n), int)  # bond type (0,1,2,3,4)
        self.m_chain = np.zeros((n, n), int)  # chain length (+N, -N, 0)
        
        occupied_sites = np.count_nonzero(self.m_bond)
        self.theta = occupied_sites / total_sites
    
    def invalidate_chains(self):
        """Mark chain cache as invalid - call when cracking happens"""
        self._chains_valid = False