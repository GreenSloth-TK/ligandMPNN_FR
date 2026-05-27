"""LigandMPNN sequence generation and structure writing."""

import json

import numpy as np
import torch


class LigandMPNNInferenceMixin:
    def generate_sequences(self, pdb_path, num_sequences=1, temperature=0.1,
                          fixed_residues=None, redesigned_residues=None):
        """
        Generate sequences using LigandMPNN with batch processing
        
        Args:
            pdb_path: Path to input PDB file
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            fixed_residues: List of residues to keep fixed
            redesigned_residues: List of residues to redesign
            
        Returns:
            List of generated sequences with scores
        """
        if self.args.verbose:
            print(f"Generating {num_sequences} sequence(s) for: {pdb_path}")
        
        # Parse PDB structure
        protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
            pdb_path,
            device=self.device,
            chains=[],  # Parse all chains
            parse_all_atoms=self.args.ligand_mpnn_use_side_chain_context or (
                self.args.pack_side_chains and not getattr(self.args, 'repack_everything', False)
            ),
            parse_atoms_with_zero_occupancy=False,
        )
        
        # Create residue encoding
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            chain_letter = chain_letters_list[i]
            icode = icodes[i] if i < len(icodes) else ""
            encoded_residue = f"{chain_letter}{R_idx_item}{icode}"
            encoded_residues.append(encoded_residue)
            
        # Set up masks for fixed/redesigned residues
        if fixed_residues is None:
            fixed_residues = []
        if redesigned_residues is None:
            redesigned_residues = []
            
        # Create chain mask
        if redesigned_residues:
            chain_mask = torch.tensor(
                [int(item in redesigned_residues) for item in encoded_residues],
                device=self.device,
            )
        elif fixed_residues:
            chain_mask = torch.tensor(
                [int(item not in fixed_residues) for item in encoded_residues],
                device=self.device,
            )
        else:
            # Design all residues
            chain_mask = torch.ones(len(encoded_residues), device=self.device)
            
        # Add chain_mask to protein_dict
        protein_dict["chain_mask"] = chain_mask
        
        # Featurize the structure (using single dict, not list)
        feature_dict = self.featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=getattr(self.args, 'ligand_mpnn_use_atom_context', True),
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )
        
        # Calculate optimal batch size
        max_batch_size = getattr(self.args, 'max_batch_size', 8)  # Adjustable max batch
        batch_size = min(num_sequences, max_batch_size)
        num_batches = (num_sequences + batch_size - 1) // batch_size  # Ceiling division
        
        if self.args.verbose:
            print(f"Processing {num_sequences} sequences in {num_batches} batches of size {batch_size}")
        
        B, L, _, _ = feature_dict["X"].shape

        omit_AA = torch.tensor(
            np.array([AA in self.args.omit_AA for AA in self.alphabet]).astype(np.float32),
            device=self.device
        )

        # Global AA bias (--bias_AA, format: "A:-1.0,P:2.3")
        bias_AA = torch.zeros(len(self.alphabet), device=self.device)
        bias_AA_str = getattr(self.args, 'bias_AA', '')
        if bias_AA_str:
            for item in bias_AA_str.split(','):
                aa, val = item.split(':')
                bias_AA[self.restype_str_to_int[aa.strip()]] = float(val)

        # Build residue index map for per-residue lookups
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))

        # Per-residue bias (--bias_AA_per_residue, JSON: {"A12": {"G": -0.3}})
        bias_AA_per_residue = torch.zeros([L, 21], device=self.device)
        bias_per_res_path = getattr(self.args, 'bias_AA_per_residue', '')
        if bias_per_res_path:
            with open(bias_per_res_path) as fh:
                bias_dict = json.load(fh)
            for res, aa_vals in bias_dict.items():
                if res in encoded_residue_dict:
                    i1 = encoded_residue_dict[res]
                    for aa, val in aa_vals.items():
                        if aa in self.alphabet:
                            bias_AA_per_residue[i1, self.restype_str_to_int[aa]] = float(val)

        # Per-residue omit (--omit_AA_per_residue, JSON: {"A12": "PG"})
        omit_AA_per_residue = torch.zeros([L, 21], device=self.device)
        omit_per_res_path = getattr(self.args, 'omit_AA_per_residue', '')
        if omit_per_res_path:
            with open(omit_per_res_path) as fh:
                omit_dict = json.load(fh)
            for res, aas in omit_dict.items():
                if res in encoded_residue_dict:
                    i1 = encoded_residue_dict[res]
                    for aa in aas:
                        if aa in self.alphabet:
                            omit_AA_per_residue[i1, self.restype_str_to_int[aa]] = 1.0

        # Set bias in feature_dict following successful pattern
        feature_dict["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )

        # Add symmetry information (empty lists)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        
        generated_sequences = []
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            current_batch_size = end_idx - start_idx
            
            # Set batch size and temperature for this batch
            feature_dict["batch_size"] = current_batch_size
            feature_dict["temperature"] = temperature
            
            # Add randn for sampling
            feature_dict["randn"] = torch.randn([current_batch_size, L], device=self.device)
            
            with torch.no_grad():
                # Sample sequences using batch processing
                output_dict = self.model.sample(feature_dict)
                
                # Get sequences and scores
                S_samples = output_dict["S"]  # Shape: [batch_size, L]
                log_probs = output_dict.get("log_probs", None)
                
                # Process each sequence in the batch
                for i in range(current_batch_size):
                    seq = self._S_to_seq(S_samples[i])
                    
                    # Calculate score if log_probs available
                    score = 0.0
                    if log_probs is not None:
                        mask = feature_dict["mask"] * feature_dict["chain_mask"]
                        # For batch processing, we need to select the right sequence
                        S_single = S_samples[i:i+1]  # Keep batch dimension
                        log_probs_single = log_probs[i:i+1]
                        score, _ = self.get_score(S_single, log_probs_single, mask)
                        score = score.item()
                        
                    generated_sequences.append({
                        'sequence': seq,
                        'score': score,
                        'S_sample': S_samples[i],
                        'temperature': temperature,
                        'index': start_idx + i,
                        'batch_idx': batch_idx
                    })
                    
                    if self.args.verbose and (start_idx + i + 1) % 5 == 0:
                        print(f"Generated sequence {start_idx + i + 1}/{num_sequences}: score {score:.4f}")
                
        if self.args.verbose:
            avg_score = np.mean([seq['score'] for seq in generated_sequences])
            print(f"Generated {len(generated_sequences)} sequences, average score: {avg_score:.4f}")
                
        return generated_sequences

    def _S_to_seq(self, S):
        """Convert sequence tensor to string"""
        return ''.join([self.restype_int_to_str[s.item()] for s in S])

    def create_structure_with_sequence(self, pdb_path, sequence, S_sample, output_path):
        """
        Create structure with new sequence using LigandMPNN's write_full_PDB
        Following run.py pattern closely for proper side chain packing
        """
        if self.args.verbose:
            print("Creating structure with new sequence using LigandMPNN...")
        
        # Parse PDB structure - use different parsing modes based on side chain packing
        parse_all_atoms_flag = (
            self.args.ligand_mpnn_use_side_chain_context or (
                self.args.pack_side_chains and not getattr(self.args, 'repack_everything', False)
            )
        )
        
        protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
            pdb_path, device=self.device, chains=[],
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=False
        )
        
        # Set the sequence in protein_dict
        protein_dict["S"] = S_sample.squeeze() if S_sample.dim() > 1 else S_sample
        if "chain_mask" not in protein_dict:
            protein_dict["chain_mask"] = torch.ones(len(protein_dict["R_idx"]), device=self.device)

        # Check if side chain packing is enabled and model is loaded
        if self.args.pack_side_chains and self.model_sc is not None:
            if self.args.verbose:
                print("Packing side chains...")
            
            # Featurize for side chain packing following run.py pattern
            feature_dict_ = self.featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=getattr(self.args, 'pack_with_ligand_context', True),
                number_of_ligand_atoms=16,
                model_type="ligand_mpnn",
            )
            
            # Prepare feature dict for side chain packing (following run.py pattern exactly)
            import copy
            sc_feature_dict = copy.deepcopy(feature_dict_)
            B = 1  # batch size for single sequence
            
            # Repeat tensors for batch processing (following run.py pattern)
            for k, v in sc_feature_dict.items():
                if k != "S":
                    try:
                        num_dim = len(v.shape)
                        if num_dim == 2:
                            sc_feature_dict[k] = v.repeat(B, 1)
                        elif num_dim == 3:
                            sc_feature_dict[k] = v.repeat(B, 1, 1)
                        elif num_dim == 4:
                            sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                        elif num_dim == 5:
                            sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                    except:
                        pass
            
            # Set the sequence for side chain packing
            # S_sample should be unsqueezed to add batch dimension
            S_for_packing = S_sample.squeeze() if S_sample.dim() > 1 else S_sample
            sc_feature_dict["S"] = S_for_packing.unsqueeze(0)  # Add batch dimension
            
            if self.args.verbose:
                print(f"Side chain packing input: S shape={sc_feature_dict['S'].shape}")
            
            # Pack side chains using LigandMPNN's pack_side_chains function
            try:
                sc_dict = self.pack_side_chains(
                    sc_feature_dict,
                    self.model_sc,
                    getattr(self.args, 'sc_num_denoising_steps', 3),
                    getattr(self.args, 'sc_num_samples', 16),
                    getattr(self.args, 'repack_everything', False),
                )
                
                if sc_dict is not None:
                    if self.args.verbose:
                        print("Side chain packing completed successfully")
                        print(f"Side chain packing output: X shape={sc_dict['X'].shape}, X_m shape={sc_dict['X_m'].shape}")
                    
                    # Extract packed coordinates
                    X_packed = sc_dict["X"]
                    X_m_packed = sc_dict["X_m"]
                    
                    # Remove batch dimension if present
                    if X_packed.dim() > 3:  # Expected: [L, 14, 3], but might be [1, L, 14, 3]
                        X_packed = X_packed.squeeze(0)
                    if X_m_packed.dim() > 2:  # Expected: [L, 14], but might be [1, L, 14]
                        X_m_packed = X_m_packed.squeeze(0)
                    
                    if self.args.verbose:
                        print(f"After squeeze: X shape={X_packed.shape}, X_m shape={X_m_packed.shape}")
                    
                    # Handle b_factors
                    if "b_factors" in sc_dict:
                        b_factors = sc_dict["b_factors"]
                        if hasattr(b_factors, 'dim') and b_factors.dim() > 2:
                            b_factors = b_factors.squeeze(0)
                        if hasattr(b_factors, 'detach'):
                            b_factors = b_factors.detach().cpu().numpy()
                        elif hasattr(b_factors, 'cpu'):
                            b_factors = b_factors.cpu().numpy()
                    else:
                        b_factors = np.ones_like(X_m_packed.detach().cpu().numpy())
                    
                    # Write full PDB using LigandMPNN's function with packed coordinates
                    self.write_full_PDB(
                        save_path=output_path,
                        X=X_packed.detach().cpu().numpy(),
                        X_m=X_m_packed.detach().cpu().numpy(),
                        b_factors=b_factors,
                        R_idx=protein_dict["R_idx"].cpu().numpy(),
                        chain_letters=protein_dict["chain_letters"],
                        S=protein_dict["S"].cpu().numpy(),
                        other_atoms=other_atoms,
                        icodes=icodes,
                        force_hetatm=getattr(self.args, 'force_hetatm', False)
                    )
                    
                    if self.args.verbose:
                        print(f"Structure with packed side chains saved: {output_path}")
                    return output_path
                else:
                    if self.args.verbose:
                        print("Side chain packing failed, falling back to backbone-only structure...")
                    # Fall through to backbone-only writing
                    
            except Exception as e:
                if self.args.verbose:
                    print(f"Warning: Side chain packing failed: {e}")
                    print("Falling back to backbone-only structure...")
                # Fall through to backbone-only writing
        else:
            # No side chain packing - use backbone coordinates like original run.py
            if self.args.verbose:
                if not self.args.pack_side_chains:
                    print("Side chain packing disabled, using backbone coordinates...")
                elif self.model_sc is None:
                    print("Side chain packer model not loaded, using backbone coordinates...")
                else:
                    print("No side chain packing, using backbone coordinates...")
        
        # Backbone-only structure creation (following run.py pattern)
        try:
            # Convert sequence to prody format following run.py pattern
            seq_prody = np.array([self.restype_1to3[AA] for AA in list(sequence)])[None,].repeat(4, 1)
            
            # Set residue names in backbone
            backbone.setResnames(seq_prody)
            
            # Set B-factors to 1.0 for all atoms
            backbone.setBetas(np.ones_like(backbone.getBetas()))
            
            # Write PDB using prody (following run.py pattern)
            if other_atoms:
                self.writePDB(output_path, backbone + other_atoms)
            else:
                self.writePDB(output_path, backbone)
            
            if self.args.verbose:
                print(f"Structure with new sequence saved: {output_path}")
            return output_path
            
        except Exception as e:
            if self.args.verbose:
                print(f"Warning: Backbone writing failed: {e}")


def parse_arguments(args=None):
    """CLI argument parser entry point."""
    from cli import parse_arguments as _parse_arguments

    return _parse_arguments(args)


