import numpy as np
from scipy import constants
from .constants import MAX_LENGTH, C_LUMP, cn
from .lattice import Lattice


class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, init_len=None,
                 k_const=None, P_H2=25, theta_H=0.61, catalyst_config=None, seed=None):
        self.temp_K          = temp_C + 273.15
        self.kb              = constants.Boltzmann
        self.reaction_time   = reaction_time
        self.P_H2            = P_H2
        self.K_H2            = 0.0158
        self.theta_H         = self.update_theta_H() if theta_H == None else float(theta_H)
        self.KB_EV = 8.617333e-5

        # Regime split points
        self.x_g = 3
        self.x_l = 11

        BASE_SHAPE = {
            # Adsorption k0 and alpha
            'ads_k0_g': 1.687976e-04, 'ads_alpha_g': +0.498165,
            'ads_k0_l': 2.308763e-03, 'ads_alpha_l': -0.729088,
            'ads_k0_h': 1.459201e-05, 'ads_alpha_h': +0.188240,
            # Desorption k0 and alpha
            'des_k0_g': 4.928056e-01, 'des_alpha_g': -3.144021,
            'des_k0_l': 4.401119e-04, 'des_alpha_l': +0.288718,
            'des_k0_h': 1.682774e-01, 'des_alpha_h': -0.090221,
            # Secondary adsorption(dMC) and cracking k
            'dmc_i': 2.061422e-02,    'dmc_t': 2.18325e-03,
            'crk_i': 9.641641e-04,    'crk_t': 1.946220e-03,
        }
        self.REGIMES = {'g': (1, self.x_g),
                        'l': (self.x_g + 1, self.x_l),
                        'h': (self.x_l + 1, MAX_LENGTH)}

        self.k_const = k_const if k_const is not None else self.build_k_const(BASE_SHAPE)

        # Build surface and initialize occupancy arrays
        self.init_active_sites(catalyst_config, seed=seed)

        if init_len is None:
            init_len = self.normal_dist(mu=280, sigma=10)

        self.init_len = init_len
        self.init_arrays(init_len)

        self.current_time   = 0.0
        self._chains        = None
        self._chains_valid  = False


    def build_k_const(self, shape: dict) -> dict:
        """Expand 16 shape parameters into the 66-key dict self.k_const holds."""
        k = {}
        for rxn in ('ads', 'des'):
            for tag, (lo, hi) in self.REGIMES.items():
                k0 = shape[f'{rxn}_k0_{tag}']
                a = shape[f'{rxn}_alpha_{tag}']
                for n in range(lo, hi + 1):
                    k[f'{rxn}_{cn(n)}'] = float(k0 * np.exp(a * n))
            k0 = shape[f'{rxn}_k0_h']
            a = shape[f'{rxn}_alpha_h']
            k[f'{rxn}_{cn(C_LUMP)}'] = float(k0 * np.exp(a * C_LUMP))
        for key in ('dmc_i', 'dmc_t', 'crk_i', 'crk_t'):
            k[key] = float(shape[key])
        return k
    
    # ------------------------------------------------------------------
    # Chain cache
    # ------------------------------------------------------------------

    @property
    def chains(self):
        """Lazy chain identification — invalidated after scission."""
        if not self._chains_valid or self._chains is None:
            self._chains       = self._identify_chains()
            self._chains_valid = True
        return self._chains

    def invalidate_chains(self):
        """Mark chain cache as stale — call after every scission event."""
        self._chains_valid = False


    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def normal_dist(self, mu=280, sigma=10, n_samples=None):
        """Draw chain length(N) from a normal distribution."""
        if n_samples is None:
            x = float(np.random.normal(loc=mu, scale=sigma))
            return int(np.floor(x))
        return np.floor(np.random.normal(loc=mu, scale=sigma, size=n_samples)).astype(int)


    # ------------------------------------------------------------------
    # Rate calculator
    # ------------------------------------------------------------------


    def get_adsorption_rate(self, chain_len):
        return self.k_const[f'ads_{cn(chain_len)}']

    def get_desorption_rate(self, chain_len):
        return self.k_const[f'des_{cn(chain_len)}']

    def get_dmc_rate(self, chain_len, is_terminal):
        if chain_len == 1: 
            return None #return None for C1 (no C-C bonds)
        return self.k_const['dmc_t' if is_terminal else 'dmc_i']

    def get_cracking_rate(self, chain_len, is_terminal):
        if chain_len == 1:
            return None #return None for C1 (no C-C bonds)
        return self.k_const['crk_t' if is_terminal else 'crk_i']

    # ------------------------------------------------------------------
    # Surface initialization
    # ------------------------------------------------------------------
 
    def init_active_sites(self, catalyst_config=None, seed=None):
        cfg = {'facet': '111', 'nx': 5, 'ny': 5, 'a': 2.77} #None -> default 5x5 (111) patch
        # if catalyst_config is not None, overwrite defaults with user-specified values. Acceptable formats:
        # 1) 8 -> int 8x8 patch, 
        # 2) {'nx': 8, 'ny': 3} -> dict 8x3 patch, 
        # 3) {'facet': '100'} -> dict 5x5 (100) patch
        if isinstance(catalyst_config, int):
            cfg.update(nx=catalyst_config, ny=catalyst_config)
        elif isinstance(catalyst_config, dict):
            cfg.update(catalyst_config)
 
        self.catalyst_config = cfg
        self.surface = Lattice(theta_H=self.theta_H, seed=seed, **cfg)
 
        # views of the lattice arrays, not copies
        self.occupancy      = self.surface.occupancy
        self.carbon_at_site = self.surface.carbon_at_site
        self.chain_at_site  = self.surface.chain_at_site
        self.h_occupancy    = self.surface.hollow_H          # read-only
        self.n_sites        = self.surface.n_sites
        self.theta_C        = self.surface.update_theta()
 
    # ------------------------------------------------------------------
    # Array initialization
    # ------------------------------------------------------------------
 
    def init_arrays(self, init_len):
        """
        carbon_array[j]   : 0=unbound, 1=bound
        chain_array[i]    : 0=break/boundary, 1=bonded  (length N+1)
        carbon_to_site[j] : bridge id holding carbon j, -1 if in the gas phase
        """
        self.carbon_array   = np.zeros(init_len, int)
 
        self.chain_array    = np.zeros(init_len + 1, int)
        self.chain_array[1:-1] = 1
 
        self.carbon_to_site = np.full(init_len, -1, int)
 
    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
 
    def update_theta(self): #Carbon
        self.theta_C = self.surface.update_theta()
        return self.theta_C

    def update_theta_H(self): #Hydrogen
        self.theta_H = np.sqrt(self.K_H2 * self.P_H2)/(1 + np.sqrt(self.K_H2 * self.P_H2))
        return self.theta_H