import numpy as np
import random

class CoverageMixin:
    """
    Tracks metal surface state using two separate matrices:
    
    m_bond:  bond type at each site
             0 = vacant
             1 = single M-C, internal carbon
             2 = single M-C, terminal carbon
             3 = double M-C, internal carbon
             4 = double M-C, terminal carbon

    m_chain: chain length at each site
             +N = first carbon of chain N (single M-C)
             -N = second carbon of same chain N (double M-C partner)
              0 = vacant
    
    Example: C7 internal adsorption → double M-C formation
        m_bond:  [..., 1, ...] → [..., 1, 3, ...]  (single → double internal)
        m_chain: [..., 7, ...] → [..., -7, -7, ...]
    """

    def metal_surface(self, reaction_key, chain_info=None):
        if reaction_key.startswith('ads_'):
            return self.coverage_adsorption(reaction_key)
        elif reaction_key.startswith('des_'):
            return self.coverage_desorption(chain_info)
        elif reaction_key.startswith('dmc_'):
            return self.coverage_dmc_formation(chain_info)
        elif reaction_key.startswith('crk_'):
            return self.coverage_cracking(chain_info)
        return False

    def update_theta(self):
        total_sites = self.m_size * self.m_size
        occupied_sites = np.count_nonzero(self.m_bond)
        self.theta = occupied_sites / total_sites
        return self.theta

    def _get_vacant_sites(self):
        return np.argwhere(self.m_bond == 0)

    def _get_neighbor_vacant(self, i, j):
        """Return vacant neighbor sites of (i,j) as list of (row, col)"""
        rows, cols = self.m_bond.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        #extract submatrix (for neighbors)
        sub = self.m_bond[row_start:row_end, col_start:col_end]
        local_vacants = np.argwhere(sub == 0)
        # Convert to global coordinates
        return [(row_start + r, col_start + c) for r, c in local_vacants]

    def _get_neighbor_by_bond(self, i, j, bond_type):
        """
        Return neighboring sites matching bond_type as list of (row, col)
        e.g., partners = self._get_neighbor_by_bond(i, j, bond_type=3)
            -> returns list; [(n_i, n_j), ...] of neighbors with double M-C internal bond
        """
        rows, cols = self.m_bond.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        sub = self.m_bond[row_start:row_end, col_start:col_end]
        local_matches = np.argwhere(sub == bond_type) 
        # bond_type can be: 
        # 1 = single M-C, internal carbon
        # 2 = single M-C, terminal carbon
        # 3 = double M-C, internal carbon
        # 4 = double M-C, terminal carbon
        return [(row_start + r, col_start + c) for r, c in local_matches]
    
    def _get_neighbor_by_chain(self, i, j, chain_value):
        """Return neighboring sites with specific m_chain value"""
        rows, cols = self.m_chain.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        
        neighbors = []
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if (r, c) != (i, j) and self.m_chain[r, c] == chain_value:
                    neighbors.append((r, c))
        return neighbors
    
    def _get_bond_type_from_adsorption(self, chain_len, ads_key):
        """Check carbon_array to see if adsorption was internal or terminal"""
        # For C1,C2, always terminal
        if chain_len <= 2:
            return 2
        
        # For C4+, check the carbon_array
        for start, end in self.chains:
            if end - start != chain_len:
                continue
            chain_segment = self.carbon_array[start:end]
            
            # Find where the 1 is
            ones = np.where(chain_segment == 1)[0]
            if len(ones) == 1:
                pos = ones[0]
                is_terminal = (pos == 0) or (pos == chain_len - 1)
                return 2 if is_terminal else 1
        
        # Fallback
        if 'internal' in ads_key:
            return 1
        return 2
    
    def _get_fragment_lengths(self, original_len, crk_key):
        """
        Determine fragment lengths after C-C cleavage.
        Uses the carbon_array cracking position for accurate split.
        """
        for start, end in self.chains:
            chain_length = end - start
            if chain_length != original_len:
                continue
            chain_segment = self.carbon_array[start:end]

            # Find the 11 pattern (double M-C bond position)
            for i in range(chain_length - 1):
                if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                    frag1 = i + 1           # Carbons 0..i
                    frag2 = chain_length - frag1  # Carbons i+1..end
                    return frag1, frag2

        # Fallback: split roughly in half
        frag1 = original_len // 2
        frag2 = original_len - frag1
        return frag1, frag2

    def _get_reacting_chain_length(self, ads_key):
        """
        For C5+ adsorption, get actual chain length from carbon_array.
        """
        is_internal = 'internal' in ads_key

        for start, end in self.chains:
            chain_length = end - start
            if chain_length < 5:
                continue
            chain_segment = self.carbon_array[start:end]
            if np.sum(chain_segment == 1) == 1: #just one carbon adsorbed
            # Return first matching chain length
                return chain_length

        return 5  # Fallback
    
    def coverage_adsorption(self, ads_key):
        # Determine chain length
        if 'c1' in ads_key:
            chain_len = 1
        elif 'c2' in ads_key:
            chain_len = 2
        elif 'c3' in ads_key:
            chain_len = 3
        elif 'c4' in ads_key:
            chain_len = 4
        else:  # c5plus
            chain_len = self._get_reacting_chain_length(ads_key)
        
        # Determine if the actual adsorption was internal or terminal
        # by looking at where the 1 appeared in carbon_array
        bond_type = self._get_bond_type_from_adsorption(chain_len, ads_key)
        
        vacant_sites = self._get_vacant_sites()
        if len(vacant_sites) == 0:
            return False
        
        row, col = random.choice(vacant_sites)
        #assign bond type(1,2,3,4) and chain length
        self.m_bond[row, col]  = bond_type
        self.m_chain[row, col] = chain_len
        
        self.update_theta()
        return True

    def coverage_desorption(self, chain_info):
        """Remove the specific chain that desorbed"""
        chain_len = chain_info
        
        # Find surface sites with this exact chain length
        candidate_sites = []
        for bond_type in [1, 2]:  # Single bonds only
            sites = np.argwhere(self.m_bond == bond_type)
            for i, j in sites:
                if self.m_chain[i, j] == chain_len:
                    candidate_sites.append((i, j))
        
        if not candidate_sites:
            return False
        
        # Pick one (or deterministically pick the first)
        i, j = candidate_sites[0]
        self.m_bond[i, j] = 0
        self.m_chain[i, j] = 0
        
        self.update_theta()
        return True
    
    def coverage_dmc_formation(self, chain_info):
        """Form double M-C bond on the specific chain"""
        chain_len = chain_info
        
        # Find sites with this exact chain length (single bonds only)
        eligible = []
        
        for bond_type in [1, 2]:
            candidate_sites = np.argwhere(self.m_bond == bond_type)
            
            for i, j in candidate_sites:
                if self.m_chain[i, j] != chain_len:
                    continue
                
                vacants = self._get_neighbor_vacant(i, j)
                if vacants:
                    eligible.append((i, j, vacants, bond_type))
        
        if not eligible:
            return False
        
        # Pick first matching (deterministic)
        i, j, vacants, source_bond = eligible[0]
        new_bond = 3 if source_bond == 1 else 4
        
        # Place second carbon
        ni, nj = vacants[0]
        self.m_bond[ni, nj]  = new_bond
        self.m_chain[i, j]   = -chain_len
        self.m_chain[ni, nj] = -chain_len
        
        self.update_theta()
        return True
    
    def coverage_cracking(self, chain_info):
        """Crack the specific chain into two fragments"""
        original_len, frag1, frag2 = chain_info
        
        # Find the double-bonded pair with this chain length
        all_negative = np.argwhere(self.m_chain < 0)
        
        for i, j in all_negative:
            if abs(self.m_chain[i, j]) != original_len:
                continue
            
            # Find partner
            neighbors = self._get_neighbor_by_chain(i, j, -original_len)
            if neighbors:
                pi, pj = neighbors[0]
                
                # Update both sites to terminal bonds with new lengths
                self.m_bond[i, j] = 2
                self.m_chain[i, j] = frag1
                
                self.m_bond[pi, pj] = 2
                self.m_chain[pi, pj] = frag2
                
                self.update_theta()
                return True
        
        return False