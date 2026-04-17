import numpy as np
from scipy import constants


class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, chain_length=None,
                 params=None, P_H2=50, catalyst_config=None):
        self.temp_K          = temp_C + 273.15
        self.kb              = constants.Boltzmann
        self.reaction_time   = reaction_time
        self.P_H2            = P_H2

        # UNIFIED PARAMETER SET (10 Core Parameters)
        default_params = {
            # 1. Adsorption/Desorption Kinetics
            'k_ads_i': 9.391672e-02,                     # rate constant for internal adsorption
            'k_ads_t': 1.077098e-04,        # rate constant for terminal adsorption
            'k_d_i'  : 9.922894e-02,        # rate constant for internal desorption
            'k_d_t'  : 6.213067e-02,        # rate constant for terminal desorption

            # 2. Van der Waals interaction scaling for gas/light/heavy species
            'alpha_vdw_gas'  : 1.262141e-02,   # vdW scaling for gas species (eV per carbon)
            'alpha_vdw_light': 1.134260e-03,   # vdW scaling for light species (eV per carbon)
            'alpha_vdw_heavy': 1.438186e-03,   # vdW scaling for heavy species (eV per carbon)

            # 3. Reaction Group (dMC & Scission)
            'k_dMC_i':  5.295269e-03,
            'k_dMC_t' : 4.058624e-03,
            'k_crk_i' : 5.619901e-04,     # dMC formation barrier (eV) (eV)
            'k_crk_t' : 9.740180e-03,     

            # 3. Hydrogen Equilibrium
            'K_H2': 5.422655e-01,        # Langmuir equilibrium constant (bar^-1)
        }
        self.params = params if params is not None else default_params

        # Build surface and initialise occupancy arrays
        self.init_active_sites(catalyst_config)

        if chain_length is None:
            chain_length = self.normal_dist(mu=280, sigma=10)

        self.chain_length = chain_length
        self.init_arrays(chain_length)

        self.current_time   = 0.0
        self._chains        = None
        self._chains_valid  = False

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

    def normal_dist(self, mu=260, sigma=30, n_samples=None):
        """Draw chain length(s) from a normal distribution."""
        if n_samples is None:
            x = float(np.random.normal(loc=mu, scale=sigma))
            return int(np.floor(x))
        return np.floor(np.random.normal(loc=mu, scale=sigma, size=n_samples)).astype(int)

    # ------------------------------------------------------------------
    # Rate calculator
    # ------------------------------------------------------------------

    def get_rate(self, N, reaction_type, is_internal=False):
        """
        Arrhenius rate for a given reaction type and chain length N.

            k_ads_i(N)    = k0_ads_i * exp(alpha_vdw * N)   # Internal adsorption rate
            k_ads_t(N)    = k0_ads_t * exp(alpha_vdw * N)   # Terminal adsorption rate

            k_d_i(N)      = k0_d_i * exp(-alpha_vdw * N)    # Internal desorption rate
            k_d_t(N)      = k0_d_t * exp(-alpha_vdw * N)    # Terminal desorption rate

            k_dMC(pos)  # dMC formation rate (terminal/internal)
            k_crk(pos)  # C-C scission rate (terminal/internal)
        """
        p   = self.params
        #kT  = 8.617e-5 * self.temp_K

        if reaction_type == 'adsorption':
            scale = ( 
                np.exp(p['alpha_vdw_gas'] * N) if N <= 4 else 
                np.exp(p['alpha_vdw_light'] * N) if N <= 12 else
                np.exp(p['alpha_vdw_heavy'] * N)
            )
            return p['k_ads_i'] * scale if is_internal else p['k_ads_t'] * scale

        elif reaction_type == 'desorption':
            scale = ( 
                np.exp(-p['alpha_vdw_gas'] * N) if N <= 4 else 
                np.exp(-p['alpha_vdw_light'] * N) if N <= 12 else
                np.exp(-p['alpha_vdw_heavy'] * N)
            )
            return p['k_d_i'] * scale if is_internal else p['k_d_t'] * scale    

        elif reaction_type == 'dMC':
            return p['k_dMC_i'] if is_internal else p['k_dMC_t']

        elif reaction_type == 'cracking':
            return p['k_crk_i'] if is_internal else p['k_crk_t']

    # ------------------------------------------------------------------
    # Array initialization
    # ------------------------------------------------------------------

    def init_arrays(self, chain_length):
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
        self.carbon_array   = np.zeros(chain_length, int)

        self.chain_array    = np.zeros(chain_length + 1, int)
        self.chain_array[1:-1] = 1

        self.hydrogen_array        = np.full(chain_length, 2, int)
        self.hydrogen_array[0]     = 3
        self.hydrogen_array[-1]    = 3

        self.carbon_to_site = np.full(chain_length, -1, int)

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
        from kmc_v3.cat_config import CatalystSurface, pt111_config

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

        # np.random.rand(n_h) generates uniform random numbers in [0,1) for each H site(n_h sites);
        # comparing with theta_H_init(e.g., 0.38) naturally gives ~38% True (occupied) and ~62% False (vacant) in h_occupancy array
        # .astype(int) converts True to 1 and False to 0, resulting in a binary occupancy array for H sites.



    #    self.theta_C        = 0.0   # already tracked via occupancy;

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def _compute_theta_H(self) -> float:
        """
        Langmuir dissociative adsorption equilibrium:
            theta_H = sqrt(K_H2 * P_H2) / (1 + sqrt(K_H2 * P_H2))

        Used once at init to seed h_occupancy.
        After init, theta_H is derived from h_occupancy directly.
        """
        K_H2 = self.params['K_H2']
        x    = np.sqrt(K_H2 * self.P_H2)
        return x / (1.0 + x)

    @property
    def theta_H(self) -> float:
        """Current H coverage fraction from explicit h_occupancy tracking."""
        return float(np.mean(self.h_occupancy)) # fraction of H sites occupied (= 0.38 for example)

    @property
    def n_vacant_h_sites(self) -> int:
        """Number of currently vacant hollow sites."""
        return int(np.sum(self.h_occupancy == 0))

    #def update_theta_C(self):
    #    """Recompute carbon coverage fraction after each reaction."""
    #    self.theta_C = np.count_nonzero(self.occupancy) / len(self.occupancy)
    #    return self.theta_C