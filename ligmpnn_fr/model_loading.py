"""LigandMPNN model loading helpers."""

import os

import torch


class ModelLoadingMixin:
    def _load_ligandmpnn_model(self):
        """Load LigandMPNN model (robust path handling like run.py)"""
        checkpoint_path = self.args.checkpoint_path
        if not checkpoint_path:
            # Handle path_to_model_weights properly
            if hasattr(self.args, 'path_to_model_weights') and self.args.path_to_model_weights:
                model_folder = self.args.path_to_model_weights
                if model_folder[-1] != '/':
                    model_folder += '/'
                # If the path doesn't end with model_params/, add it
                if not model_folder.endswith('model_params/'):
                    model_folder += 'model_params/'
            else:
                model_folder = './model_params/'
            checkpoint_path = f"{model_folder}{getattr(self.args, 'model_name', 'ligandmpnn_v_32_010_25')}.pt"
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("Could not determine checkpoint_path for LigandMPNN model.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        if self.args.verbose:
            print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        atom_context_num = checkpoint.get("atom_context_num", 1)
        k_neighbors = checkpoint["num_edges"]
        model = self.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type="ligand_mpnn",
            ligand_mpnn_use_side_chain_context=self.args.ligand_mpnn_use_side_chain_context,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_packer_model(self):
        """Load LigandMPNN side chain packer model following run.py pattern"""
        # Set default checkpoint path if not provided
        checkpoint_path_sc = getattr(self.args, 'checkpoint_path_sc', None)
        
        if not checkpoint_path_sc:
            # Use default path structure like run.py
            model_folder = getattr(self.args, 'path_to_model_weights', None) or os.environ.get('LMPNN_DIR', '')
            if model_folder[-1] != '/':
                model_folder += '/'
            checkpoint_path_sc = f"{model_folder}ligandmpnn_sc_v_32_002_16.pt"
        
        # Validate checkpoint path
        if not checkpoint_path_sc or not isinstance(checkpoint_path_sc, str):
            if self.args.verbose:
                print("Invalid checkpoint path for side chain packer. Packing disabled.")
            return None
            
        if not os.path.exists(checkpoint_path_sc):
            if self.args.verbose:
                print(f"Side chain packer model not found: {checkpoint_path_sc}")
                print("Side chain packing will be disabled")
            return None
            
        if self.args.verbose:
            print(f"Loading side chain packer from: {checkpoint_path_sc}")
            
        try:
            checkpoint_sc = torch.load(checkpoint_path_sc, map_location=self.device, weights_only=True)
            
            # Create side chain packer model following run.py exactly
            model_sc = self.Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=self.device,
                num_mix=3,
            )
            
            model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
            model_sc.to(self.device)
            model_sc.eval()
            
            return model_sc
            
        except Exception as e:
            if self.args.verbose:
                print(f"Error loading side chain packer: {e}")
                print("Side chain packing will be disabled")
            return None
