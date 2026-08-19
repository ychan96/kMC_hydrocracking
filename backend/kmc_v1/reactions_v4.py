import numpy as np
import random
from scipy import stats
from scipy.stats import skewnorm
from collections import Counter, defaultdict

class ReactionMixin:
    """
    Handles reaction selection and execution for kinetic Monte Carlo simulations.
    Key scheme:
    - ads/des: single key per chain length (ads_cN / des_cN); C31+ catch-all
    - dMC / cracking: length-integrated  →  dmc_i / dmc_t / crk_i / crk_t
    """
    
    def select_reaction(self, site_counts):
        rates = {}

        # Adsorption — C2+; C31+ catch-all
        ads_keys = [f'ads_c{n}' for n in range(2, 31)] + ['ads_c31plus']

        for key in ads_keys:
            count = site_counts.get(key, 0)
            if count > 0:
                rates[key] = self.k_const[key] * count * (1 - self.theta)

        # Desorption — C1+; C31+ catch-all
        des_keys = [f'des_c{n}' for n in range(1, 31)] + ['des_c31plus']

        for key in des_keys:
            count = site_counts.get(key, 0)
            if count > 0:
                rates[key] = self.k_const[key] * count

        # dMC — length-integrated
        for key in ['dmc_i', 'dmc_t']:
            count = site_counts.get(key, 0)
            if count > 0:
                rates[key] = self.k_const[key] * count

        # Cracking — length-integrated
        for key in ['crk_i', 'crk_t']:
            count = site_counts.get(key, 0)
            if count > 0:
                rates[key] = self.k_const[key] * count
        
        # Total rate
        R = sum(rates.values())
        
        # No reactions possible
        if R == 0:
            return None, 0
        
        # Select reaction using KMC algorithm
        u1 = np.random.rand()
        u2 = np.random.rand()
        
        # Create cumulative distribution
        reaction_keys = list(rates.keys())
        rate_values = np.array([rates[k] for k in reaction_keys])
        cdf = np.cumsum(rate_values) / R #normalization of getting CDF [p1,p1+p2,...,1]
        
        # idx = np.argmax(cdf >= u1) 
        # cdf < u1 gives a True up to the last bin like cdf=[0.2, 0.5, 0.7, 1.0], u1 = 0.65 -> cdf < u1 [T,T,F,F] where always returns the 0(first T=1)
        # Select reaction: find the FIRST index where cdf >= u1
        idx = np.searchsorted(cdf, u1, side='right') 
        selected_reaction = reaction_keys[idx]
        
        # Time increment
        dt = -np.log(u2) / R 
        
        return selected_reaction, dt #reaction_key, dt
    
    def perform_reaction(self, reaction_key):
        """
        Execute the selected reaction.
        
        Args:
            reaction_key (str): Reaction identifier (e.g., 'ads_c1', 'crk_c4_internal')
        
        Returns:
            bool: True if reaction succeeded, False otherwise
        """
        # Parse reaction type
        if reaction_key.startswith('ads_'):
            return self.perform_adsorption(reaction_key)
        elif reaction_key.startswith('des_'):
            return self.perform_desorption(reaction_key)
        elif reaction_key.startswith('dmc_'):
            return self.perform_dmc_formation(reaction_key)
        elif reaction_key.startswith('crk_'):
            return self.perform_cracking(reaction_key)
        else:
            return False, None
        
    def perform_adsorption(self, ads_key, use_normal=False):
        """Perform adsorption on vacant sites."""
        if 'c31plus' in ads_key:
            target_length = 31
        else:
            target_length = int(ads_key.split('_c')[1])

        adsorption_sites = []

        for start, end in self.chains:
            chain_length  = end - start
            chain_segment = self.carbon_array[start:end]

            if target_length < 31:
                if chain_length != target_length:
                    continue
            else:
                if chain_length < 31:
                    continue

            if np.any(chain_segment == 1):
                continue

            # All vacant positions are candidates
            candidates = [start + i for i in range(chain_length) if chain_segment[i] == 0]
            if candidates:
                adsorption_sites.extend(candidates)

        if not adsorption_sites:
            return False, None

        site = random.choice(adsorption_sites)
        self.carbon_array[site] = 1
        return True, None
    
    def _get_chain_info_for_carbon(self, carbon_idx):
        """
        Get chain length that this carbon belongs to
        self.chains comes from the @property in init.py -> it calls _identify_chains() from count_sites.py
        """
        for start, end in self.chains: 
            if start <= carbon_idx < end:
                return end - start
        return None

    def perform_desorption(self, des_key):
        """Perform desorption on a single attached carbon."""
        if 'c31plus' in des_key:
            target_length = 31
        else:
            target_length = int(des_key.split('_c')[1])

        desorption_sites = []

        for start, end in self.chains:
            chain_length  = end - start
            chain_segment = self.carbon_array[start:end]

            if target_length < 31:
                if chain_length != target_length:
                    continue
            else:
                if chain_length < 31:
                    continue

            # Exactly one attached carbon
            if np.sum(chain_segment == 1) != 1:
                continue

            attached_idx = np.where(chain_segment == 1)[0][0]
            desorption_sites.append(start + attached_idx)

        if not desorption_sites:
            return False, None

        site       = random.choice(desorption_sites)
        chain_info = self._get_chain_info_for_carbon(site)
        self.carbon_array[site] = 0
        return True, chain_info


    
    def perform_dmc_formation(self, dmc_key):
        """
        Perform double M-C bond formation (dehydrogenation).
        dmc_key: 'dmc_i' or 'dmc_t'
        """
        is_internal = dmc_key.endswith('_i')

        dmc_sites = []

        for start, end in self.chains:
            chain_length  = end - start
            chain_segment = self.carbon_array[start:end]

            if chain_length == 1:
                continue

            if np.sum(chain_segment == 1) != 1:
                continue

            attached_idx = np.where(chain_segment == 1)[0][0]
            left_vacant  = attached_idx > 0 and chain_segment[attached_idx - 1] == 0
            right_vacant = attached_idx < chain_length - 1 and chain_segment[attached_idx + 1] == 0

            if not (left_vacant or right_vacant):
                continue

            # C2, C3: terminal only
            if chain_length in (2, 3):
                if not is_internal:
                    if left_vacant:
                        dmc_sites.append(start + attached_idx - 1)
                    if right_vacant:
                        dmc_sites.append(start + attached_idx + 1)
            else:  # C4+
                if left_vacant:
                    bond_at_end = (attached_idx - 1 == 0) or (attached_idx == chain_length - 1)
                    if is_internal != bond_at_end:
                        dmc_sites.append(start + attached_idx - 1)
                if right_vacant:
                    bond_at_end = (attached_idx == 0) or (attached_idx == chain_length - 2)
                    if is_internal != bond_at_end:
                        dmc_sites.append(start + attached_idx + 1)

        if not dmc_sites:
            return False, None

        site       = random.choice(dmc_sites)
        chain_info = self._get_chain_info_for_carbon(site)
        self.carbon_array[site] = 1

        return True, chain_info
    
    def perform_cracking(self, crk_key):
        """
        Perform C-C bond scission.
        crk_key: 'crk_i' or 'crk_t'
        """
        is_internal = crk_key.endswith('_i')

        cracking_sites = []

        for start, end in self.chains:
            chain_length  = end - start
            chain_segment = self.carbon_array[start:end]

            if chain_length == 1:
                continue

            for i in range(chain_length - 1):
                if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                    bond_at_end = (i == 0) or (i == chain_length - 2)
                    if is_internal != bond_at_end:
                        cracking_sites.append(start + i + 1)

        if not cracking_sites:
            return False, None

        chain_index = random.choice(cracking_sites)

        # Get fragment info before breaking
        for start, end in self.chains:
            if start < chain_index <= end:
                original_len  = end - start
                chain_segment = self.carbon_array[start:end]
                for i in range(len(chain_segment) - 1):
                    if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                        frag1 = i + 1
                        frag2 = original_len - frag1
                        break
                break

        self.chain_array[chain_index] = 0
        self.invalidate_chains()
        chain_info = (original_len, frag1, frag2)

        return True, chain_info