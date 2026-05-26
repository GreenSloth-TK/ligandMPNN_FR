"""Runtime setup helpers for LigandMPNN and PyRosetta."""

import os
import sys


def setup_ligandmpnn():
    """Setup LigandMPNN model and dependencies"""
    # Add LigandMPNN to path
    ligandmpnn_path = os.environ.get('LMPNN_DIR', '/apps/repos/LigandMPNN')
    if ligandmpnn_path not in sys.path:
        sys.path.insert(0, ligandmpnn_path)
    
    try:
        from data_utils import (
            alphabet,
            featurize,
            get_score,
            parse_PDB,
            restype_int_to_str,
            restype_str_to_int,
            restype_1to3,
            write_full_PDB,
        )
        from model_utils import ProteinMPNN
        from sc_utils import Packer, pack_side_chains
        from prody import writePDB
        return True, (parse_PDB, featurize, get_score, alphabet, restype_int_to_str, restype_str_to_int, restype_1to3, ProteinMPNN, write_full_PDB, Packer, pack_side_chains, writePDB)
    except ImportError as e:
        print(f"Error importing LigandMPNN modules: {e}")
        return False, None


def setup_pyrosetta(ligand_params_path, native_pdb_path, use_genpot=False, verbose=True):
    """Setup PyRosetta with appropriate flags"""
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import get_score_function
        from pyrosetta import toolbox
        
        # Build initialization flags
        init_flags = ['-beta']
        if not verbose:
            init_flags.append('-mute all')
        
        # Improved multithreading configuration for main process
        # Use environment variables to determine thread count
        pyrosetta_threads = int(os.environ.get('ROSETTA_NUM_THREADS', '4'))
        init_flags.extend([
            f'-multithreading:total_threads {pyrosetta_threads}',
            f'-multithreading:interaction_graph_threads {pyrosetta_threads}'
        ])
        
        if use_genpot:
            init_flags.extend([
                '-gen_potential',
                f'-extra_res_fa {ligand_params_path}',
                f'-in:file:native {native_pdb_path}'
            ])
        else:
            init_flags.extend([
                f'-extra_res_fa {ligand_params_path}',
                f'-in:file:native {native_pdb_path}'
            ])
        
        if verbose:
            print("Initializing PyRosetta...")
        pyrosetta.init(' '.join(init_flags))
        
        return True, (pyrosetta, FastRelax, get_score_function, toolbox)
    
    except ImportError as e:
        print(f"Error importing PyRosetta: {e}")
        return False, None
