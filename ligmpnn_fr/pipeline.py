"""Iterative LigandMPNN + FastRelax pipeline assembly."""

import torch

from .inference import LigandMPNNInferenceMixin
from .model_loading import ModelLoadingMixin
from .mpnn_setup import setup_ligandmpnn, setup_pyrosetta
from .relaxation import RelaxationMixin
from .selection import SelectionMixin
from .workflow import WorkflowMixin


class LigandMPNNFastRelax(
    ModelLoadingMixin,
    LigandMPNNInferenceMixin,
    RelaxationMixin,
    SelectionMixin,
    WorkflowMixin,
):
    """Pipeline coordinator for iterative LigandMPNN design and FastRelax."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup LigandMPNN
        success, modules = setup_ligandmpnn()
        if not success:
            raise RuntimeError("Failed to setup LigandMPNN")
        
        self.parse_PDB, self.featurize, self.get_score, self.alphabet, self.restype_int_to_str, self.restype_str_to_int, self.restype_1to3, self.ProteinMPNN, self.write_full_PDB, self.Packer, self.pack_side_chains, self.writePDB = modules
        
        # Load model
        self.model = self._load_ligandmpnn_model()
        
        # Load side chain packing model if enabled
        self.model_sc = None
        if getattr(args, 'pack_side_chains', False):
            try:
                if args.verbose:
                    print("Loading side chain packing model...")
                self.model_sc = self._load_packer_model()
                if self.model_sc is not None:
                    if args.verbose:
                        print("Side chain packing model loaded successfully!")
                else:
                    if args.verbose:
                        print("Side chain packing model failed to load!")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: Could not load side chain packer: {e}")
                    print("Side chain packing will be disabled")
                self.model_sc = None
        else:
            if args.verbose:
                print("Side chain packing is disabled (pack_side_chains=False)")
        
        # Setup PyRosetta
        success, modules = setup_pyrosetta(
            args.ligand_params_path, 
            args.pdb_path, 
            args.use_genpot,
            args.verbose
        )
        if not success:
            raise RuntimeError("Failed to setup PyRosetta")
        
        self.pyrosetta, self.FastRelax, self.get_score_function, self.toolbox = modules
        
        # Setup output directories
        self._setup_output_dirs()
