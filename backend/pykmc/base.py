import numpy as np
from scipy import constants
from .constants import MAX_LENGTH, C_LUMP, cn


class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, init_len=None,
                 k_const=None, P_H2=50, catalyst_config=None):
        self.temp_K          = temp_C + 273.15
        self.kb              = constants.Boltzmann
        self.reaction_time   = reaction_time
        self.P_H2            = P_H2
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

        SHAPE_KEYS = list(BASE_SHAPE.keys())
        FAMILIES = {'ads': ['ads_k0_g', 'ads_k0_l', 'ads_k0_h'],
                    'des': ['des_k0_g', 'des_k0_l', 'des_k0_h'],
                    'dmc': ['dmc_i', 'dmc_t'],
                    'crk': ['crk_i', 'crk_t']}
        self.REGIMES = {'g': (1, self.x_g),
                        'l': (self.x_g + 1, self.x_l),
                        'h': (self.x_l + 1, MAX_LENGTH)}

        self.k_const = k_const if k_const is not None else self.build_k_const(BASE_SHAPE)

        # Constant for the whole run — depends only on P_H2
        self.eff_H = self.eff_H_availability(self.P_H2)

        # Build surface and initialise occupancy arrays
        self.init_active_sites(catalyst_config)

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
        """Draw chain length(s) from a normal distribution."""
        if n_samples is None:
            x = float(np.random.normal(loc=mu, scale=sigma))
            return int(np.floor(x))
        return np.floor(np.random.normal(loc=mu, scale=sigma, size=n_samples)).astype(int)


    # ------------------------------------------------------------------
    # Rate calculator
    # ------------------------------------------------------------------
    
    def eff_H_availability(self, P_H2):
        """Effective surface H availability at a given H2 pressure."""
        #Heuristic fit to H2 pressure dependence
        p_c = 13.75
        n = 1.164
        return 1-np.exp(-(P_H2/p_c)**n)

    def get_adsorption_rate(self, chain_len, is_terminal=None):
        return self.k_const[f'ads_{cn(chain_len)}']

    def get_desorption_rate(self, chain_len, is_terminal=None):
        return self.k_const[f'des_{cn(chain_len)}']

    def get_dmc_rate(self, chain_len, is_terminal):
        """Returns None for C1 (no dMC reaction)."""
        if chain_len == 1:
            return None
        if is_terminal:
            rate = self.k_const['dmc_t'] * self.eff_H 
        else:
            rate = self.k_const['dmc_i'] * self.eff_H
        return rate

    def get_cracking_rate(self, chain_len, is_terminal):
        """Returns None for C1 (no C-C bonds)."""
        if chain_len == 1:
            return None
        if is_terminal:
            rate = self.k_const['crk_t'] * self.eff_H
        else:
            rate = self.k_const['crk_i'] * self.eff_H
        return rate

    # ------------------------------------------------------------------
    # Array initialization
    # ------------------------------------------------------------------

    def init_arrays(self, init_len):
        """
        Initialise per-carbon tracking arrays.

        carbon_array[j]   : 0=free, 1=adsorbed
        chain_array[i]    : 0=break/boundary, 1=bonded  (length N+1)
        hydrogen_array[j] : H count on carbon j
                            3 = free terminal carbon
                            2 = free internal carbon
                            0 = adsorbed (H released to surface)

        carbon_to_site[j] : surface ATOP site index holding carbon j(global_c) (-1 if free)
        """
        self.carbon_array   = np.zeros(init_len, int)

        self.chain_array    = np.zeros(init_len + 1, int)
        self.chain_array[1:-1] = 1

        self.hydrogen_array        = np.full(init_len, 2, int)
        self.hydrogen_array[0]     = 3
        self.hydrogen_array[-1]    = 3

        self.carbon_to_site = np.full(init_len, -1, int)


    def init_active_sites(self, catalyst_config=None):
        """
        Build the CatalystSurface and initialise all surface-state arrays.

        C-site arrays  (length = n_c_sites):
            occupancy[i]      : 0=vacant, 1=single M-C, 2=dMC
            chain_at_site[i]  : fragment length of carbon at site i (0 if vacant) // negative if dMC (e.g., {-3,-3} for a dMC from a C3 fragment)
            carbon_at_site[i] : carbon_array index of carbon at site i (-1 if vacant)

        H-site array  (length = n_h_sites):
            h_occupancy[i]    : 0=vacant, 1=occupied
            Initialised by randomly populating hollow sites according to
            theta_H_init = sqrt(K_H2 * P_H2) / (1 + sqrt(K_H2 * P_H2)).
            After init, every adsorption/desorption event updates h_occupancy
            explicitly via hydrogen_array changes.
        """
        from .lattice import CatalystSurface, pt111_config

        if catalyst_config is None:
            catalyst_config = pt111_config()

        self.surface        = CatalystSurface(catalyst_config)
        n_c                 = self.surface.n_c_sites
        n_h                 = self.surface.n_h_sites #length of n_h_sites = number of hollow sites

        # C-site state
        self.occupancy      = np.zeros(n_c, int)
        self.chain_at_site  = np.zeros(n_c, int)
        self.carbon_at_site = np.full(n_c, -1, int)

        # H-site state — seeded from Langmuir equilibrium at t=0
        theta_H_init        = self._compute_theta_H()
        self.h_occupancy    = (np.random.rand(n_h) < theta_H_init).astype(int) 