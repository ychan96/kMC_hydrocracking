import numpy as np
import random
from scipy import stats
from scipy.stats import skewnorm
from collections import Counter, defaultdict

class ReactionMixin:
    """
    Reaction selection using Gillespie algorithm.
    Consumes update_configuration() output:
        counts['adsorption'][N]['terminal']   -> int
        counts['adsorption'][N]['internal']   -> int
        counts['desorption'][N]['terminal']   -> int
        counts['desorption'][N]['internal']   -> int
        counts['dmc'][N]['terminal']          -> int
        counts['dmc'][N]['internal']          -> int
        counts['cracking'][N]['terminal']     -> int
        counts['cracking'][N]['internal']     -> int

    Arrays
        carbon_array[i]: i-th carbon = 0 or 1
        hydrogen_array[i]: # of H atoms on i-th carbon 3 or 2

    """

    def select_reaction(self, counts):
        rates = {}

        # Adsorption — k_ads(N) = k0,ads * exp(alpha_vdw * N)
        for N, pos_counts in counts['adsorption'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('adsorption', N, pos)] = self.get_rate(N, 'adsorption', is_internal) * n_sites

        # Desorption — k_d(N) = k0,d * exp(-alpha_vdw * N)
        for N, pos_counts in counts['desorption'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('desorption', N, pos)] = self.get_rate(N, 'desorption', is_internal) * n_sites

        # dMC formation — k_dMC(pos) = is_internal ? k_dMC_i : k_dMC_t
        for N, pos_counts in counts['dmc'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('dmc', N, pos)] = self.get_rate(N, 'dMC', is_internal) * n_sites

        # Cracking — k_crk(pos) = is_internal ? k_crk_i : k_crk_t
        for N, pos_counts in counts['cracking'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('cracking', N, pos)] = self.get_rate(N, 'cracking', is_internal) * n_sites

        R = sum(rates.values())
        if R == 0:
            return None, 0

        # BKL selection
        reaction_keys = list(rates.keys())
        rate_values   = np.array([rates[k] for k in reaction_keys])
        cum           = np.cumsum(rate_values) / R

        u1 = np.random.rand()
        u2 = np.random.rand()

        idx               = np.searchsorted(cum, u1, side='right')
        selected_reaction = reaction_keys[idx]       # (reaction_type, N, pos)

        dt = -np.log(u2) / R

        return selected_reaction, dt
    
    def perform_reaction(self, reaction_key):
        reaction_type, N, pos = reaction_key

        if reaction_type == 'adsorption':
            return self.perform_adsorption(N, pos)
        elif reaction_type == 'desorption':
            return self.perform_desorption(N, pos)
        elif reaction_type == 'dmc':
            return self.perform_dmc_formation(N, pos)
        elif reaction_type == 'cracking':
            return self.perform_cracking(N, pos)
        else:
            return False
        
    def sample_adsorption_site(self, free_positions, chain_start, chain_length, use_normal=True):
        if not free_positions:
            return None

        if use_normal:
            mid     = chain_start + (chain_length - 1) / 2 #global
            sigma   = chain_length / 8
            weights = stats.norm.pdf(free_positions, loc=mid, scale=sigma) #calculates probability density at each free position
            weights /= weights.sum() #normalize to sum to 1
            return int(np.random.choice(free_positions, p=weights)) #picks one position from normal distribution centered on the middle of the chain, with preference for central sites
        else:
            return int(np.random.choice(free_positions)) #picks one position uniformly
    
    def perform_adsorption(self, N, pos):
        # 1. Find all N-length fragments and add start idx
        candidate_fragments = []
        for start, end in self.chains:
            if end - start == N:
                seg = self.carbon_array[start:end]
                if not np.any(seg == 1):
                    candidate_fragments.append(start)

        if not candidate_fragments:
            return False

        # 2. Randomly pick one N fragment(start idx)
        start = np.random.choice(candidate_fragments)
        seg   = self.carbon_array[start:start + N]

        # 3. Pick one carbon in the N fragment
        if pos == 'terminal':
            local_c = int(np.random.choice([0, N - 1]))
            global_c = start + local_c
        else:
            internal_positions = list(np.where(seg[1:-1] == 0)[0] + 1 + start) #global
            global_c = self.sample_adsorption_site(internal_positions, start, N, use_normal=True)
            local_c = global_c - start
        
        is_terminal = (pos == 'terminal')


        # 4. Find a vacant atop site(C) on the surface
        vacant_c_sites = np.where(self.occupancy == 0)[0]
        if len(vacant_c_sites) == 0:
            return False
        site_idx = np.random.choice(vacant_c_sites) #physical site index on the surface

        # 5. Find a vacant hollow site(H) for released H atoms — check availability before making any changes
        n_h_released   = 3 if is_terminal else 2
        vacant_h_sites = np.where(self.h_occupancy == 0)[0]
        if len(vacant_h_sites) < n_h_released:
            return False   # not enough H sites — should be gated in count_sites

        # 6. Update carbon arrays
        self.carbon_array[global_c]    = 1
        self.hydrogen_array[global_c]  = 0   # 3->0 terminal, 2->0 internal

        # 7. Update C site arrays
        self.occupancy[site_idx]       = 1        # site_idx is now single M-C
        self.carbon_at_site[site_idx]  = global_c # global_c(carbon_array) -> site_idx(occupancy array)
        self.chain_at_site[site_idx]   = N        # which chain is the owner of this site_idx
        self.carbon_to_site[global_c]  = site_idx # site_idx(occupancy array) -> global_c(carbon_array)

        # 8. Scatter released H atoms onto vacant hollow sites
        chosen_h_sites = np.random.choice(vacant_h_sites, size=n_h_released, replace=False) #no duplicates
        self.h_occupancy[chosen_h_sites] = 1
        
        return True
    
    def perform_desorption(self, N, pos):
        # 1. Find all single-MC carbons in length-N fragments
        candidate_carbons = []
        for start, end in self.chains:
            if end - start != N: #chain-length matching
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1: #is it adsorbed??
                continue
            local_pos  = np.where(seg == 1)[0][0] #accesses the first index array and first index
            global_pos = start + local_pos
            site_idx   = self.carbon_to_site[global_pos]
            #sanity check — is this carbon actually adsorbed at a single-MC site?
            if site_idx == -1 or self.occupancy[site_idx] != 1:
                continue
            #match pos from reaction_key
            is_terminal = (local_pos == 0) or (local_pos == N - 1)
            if (pos == 'terminal') != is_terminal:
                continue
            candidate_carbons.append((global_pos, local_pos))

        if not candidate_carbons:
            return False

        # 2. Pick one randomly
        global_c, local_c = candidate_carbons[np.random.choice(len(candidate_carbons))]
        is_terminal = (pos == 'terminal')
        site_idx    = self.carbon_to_site[global_c] #find surface site

        # 3. check H availability before making any changes
        n_h_returned    = 3 if is_terminal else 2
        occupied_h      = np.where(self.h_occupancy == 1)[0]
        if n_h_returned > len(occupied_h):
            return False   # not enough H to return — should be gated in count_sites

        # 4. Update carbon arrays
        self.carbon_array[global_c]   = 0
        self.hydrogen_array[global_c] = 3 if is_terminal else 2  # restore H count

        # 5. Update ATOP site arrays
        self.occupancy[site_idx]      = 0   #no bond
        self.carbon_at_site[site_idx] = -1  #Carbon index is free => -1
        self.chain_at_site[site_idx]  = 0   #no chain occupies this site 
        self.carbon_to_site[global_c] = -1  #ATOP site is free => -1

        # 6. Return H atoms — free random hollow sites
        chosen_h_sites  = np.random.choice(occupied_h, size=n_h_returned, replace=False) #(pool, how many, no site picked twice)
        self.h_occupancy[chosen_h_sites] = 0

        return True
    
    def perform_dmc_formation(self, N, pos):
        # 1. Find eligible fragments — exactly one single-MC carbon with a free neighbor
        candidate_pairs = []
        for start, end in self.chains:
            if end - start != N:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1:
                continue

            local_pos = np.where(seg == 1)[0][0]
            global_pos = start + local_pos
            site_idx = self.carbon_to_site[global_pos]
            if site_idx == -1 or self.occupancy[site_idx] != 1:
                continue

            # Check vacant surface neighbor exists
            if not any(self.occupancy[nb] == 0 for nb in self.surface.get_c_neighbors(site_idx)): #get_c_neighbors -> list of adjacent ATOP sites
                continue #fully occupied neigbors

            for nb_pos in [local_pos - 1, local_pos + 1]:
                if nb_pos < 0 or nb_pos >= N:
                    continue
                if seg[nb_pos] != 0:
                    continue
                at_terminal = (local_pos == 0 or local_pos == N-1 or nb_pos == 0 or nb_pos == N-1)
                if (pos == 'internal') == (not at_terminal):
                    candidate_pairs.append((global_pos, start + nb_pos, start))

        if not candidate_pairs:
            return False

        # 2. Pick one pair (global, global-1) or (global, global+1)
        global_c, global_nb_c, start = candidate_pairs[np.random.choice(len(candidate_pairs))] #global
        anchor_idx = self.carbon_to_site[global_c]

        # 3. Find vacant surface neighbor for global_nb_c
        vacant_nb = [nb for nb in self.surface.get_c_neighbors(anchor_idx)
                    if self.occupancy[nb] == 0]
        nb_idx  = np.random.choice(vacant_nb)

        # 4. Check H avaulability
        local_nb_c = global_nb_c - start
        is_terminal = (local_nb_c == 0) or (local_nb_c == N - 1)
        n_h_released = 3 if is_terminal else 2
        vacant_h_sites = np.where(self.h_occupancy == 0)[0]
        if len(vacant_h_sites) < n_h_released:
            return False   # not enough H sites — should be gated in count_sites

        # 5. Update carbon arrays
        self.carbon_array[global_nb_c]   = 1
        self.hydrogen_array[global_nb_c] = 0

        # 6. Update ATOP site arrays
        for site_idx in [anchor_idx, nb_idx]:
            self.occupancy[site_idx]      = 2        
            self.chain_at_site[site_idx]  = -N  # -N pair -> dMC from N fragment

        self.carbon_at_site[nb_idx] = global_nb_c
        self.carbon_to_site[global_nb_c]  = nb_idx

        # 7. Release H atoms
        chosen_h = np.random.choice(vacant_h_sites, size=n_h_released, replace=False)
        self.h_occupancy[chosen_h] = 1

        return True

    def perform_cracking(self, N, pos):
        # 1. Find eligible 11 patterns
        candidate_bonds = []
        for start, end in self.chains:
            if end - start != N:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 2:
                continue

            for i in range(N - 1):
                if seg[i] == 1 and seg[i + 1] == 1:
                    at_terminal = (i == 0) or (i == N - 2)
                    if (pos == 'internal') == (not at_terminal):
                        candidate_bonds.append((start, end, start + i + 1))  # chain_array index

        if not candidate_bonds:
            return False

        # 2. Pick one bond
        start, end, chain_idx = candidate_bonds[np.random.choice(len(candidate_bonds))]
        seg = self.carbon_array[start:end]
        local_c_l = chain_idx - start - 1  # local position of left carbon

        # 3. Get the two carbons
        global_c_l = start + local_c_l
        global_c_r = start + local_c_l + 1
        l_idx  = self.carbon_to_site[global_c_l]
        r_idx  = self.carbon_to_site[global_c_r]

        # 4. Update ATOP site arrays — two fragments are formed *no change in carbon array
        left_N = chain_idx - start
        right_N = end - chain_idx

        self.occupancy[l_idx] = 1              #2->1: dMC -> single MC
        self.chain_at_site[l_idx] = left_N     #dMC -> single MC
        self.occupancy[r_idx] = 1              #2->1: dMC -> single MC
        self.chain_at_site[r_idx] = right_N    #dMC -> single MC

        # 5. Break chain bond
        self.chain_array[chain_idx] = 0

        self.invalidate_chains()
        return True
    