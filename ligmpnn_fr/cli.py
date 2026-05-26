"""Command-line interface for iterative LigandMPNN-FastRelax."""

import multiprocessing as mp
import random

import numpy as np
import torch

from .pipeline import LigandMPNNFastRelax


def parse_arguments(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='LigandMPNN FastRelax - Complete Version')
    
    # Input/Output
    parser.add_argument('--pdb_path', type=str, required=True,
                        help='Path to input PDB file')
    parser.add_argument('--ligand_params_path', type=str, required=True,
                        help='Path to ligand parameters file')
    parser.add_argument('--out_folder', type=str, default='./ligmpnn_fr_output',
                        help='Output folder path')
    
    # Model settings
    import os
    default_weights = os.path.join(os.environ.get('LMPNN_DIR', ''), 'model_params') or None
    parser.add_argument('--path_to_model_weights', type=str, default=default_weights,
                        help='Path to model weights (defaults to $LMPNN_DIR/model_params)')
    parser.add_argument('--model_name', type=str, default='ligandmpnn_v_32_010_25',
                        help='Model name')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Direct path to checkpoint file')
    
    # Sequence generation
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--num_seq_per_target', type=int, default=1,
                        help='Number of sequences per target')
    
    # Iterative design
    parser.add_argument('--n_cycles', type=int, default=3,
                        help='Number of iterative design cycles')
    
    # Constraints and design control
    parser.add_argument('--fixed_residues', type=str, nargs='*', default=[],
                        help='List of residues to keep fixed')
    parser.add_argument('--redesigned_residues', type=str, nargs='*', default=[],
                        help='List of residues to redesign')
    parser.add_argument('--omit_AAs', type=str, default='X',
                        help='Amino acids to omit')
    
    # PyRosetta settings
    parser.add_argument('--use_genpot', action='store_true',
                        help='Use genpot for fast relax')
    
    # XML relaxation settings (from original ligMPNN_FR)
    parser.add_argument('--repackable_res', type=str, default='',
                        help='Repackable residue numbers concatenated with comma (e.g., "10,15,27")')
    parser.add_argument('--target_atm_for_cst', type=str, default='',
                        help='Target ligand atom names to extract distance constraints from input design (e.g., "O,N")')
    
    # LigandMPNN settings
    parser.add_argument('--use_side_chain_context', action='store_true',
                        help='Use side chain context in LigandMPNN')
    parser.add_argument('--ligand_mpnn_use_side_chain_context', type=int, default=0,
                        help='Use side chain context in LigandMPNN')
    parser.add_argument('--ligand_mpnn_use_atom_context', type=int, default=1,
                        help='Use atom context in LigandMPNN')
    parser.add_argument('--pack_side_chains', action='store_true', default=False,
                        help='Use LigandMPNN side chain packing')
    parser.add_argument('--checkpoint_path_sc', type=str, default=None,
                        help='Path to side chain packer model')
    parser.add_argument('--pack_with_ligand_context', type=int, default=1,
                        help='Use ligand context during side chain packing')
    
    # Batch processing and parallelization
    parser.add_argument('--max_batch_size', type=int, default=1,
                        help='Maximum batch size for sequence generation (default: 1)')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes for fast relax (default: 1)')
    parser.add_argument('--pyrosetta_threads', type=int, default=1,
                        help='Number of threads per PyRosetta process (default: 1)')
    parser.add_argument('--relax_mode', type=str, default='fastrelax',
                        choices=['fastrelax', 'score_only'],
                        help='Relaxation mode: full FastRelax or score-only smoke test')
    parser.add_argument('--relax_timeout', type=int, default=0,
                        help='Seconds before a relaxation worker is terminated; 0 disables timeout')

    # Side chain packing parameters
    parser.add_argument('--number_of_packs_per_design', type=int, default=1,
                        help='Number of side chain packing samples per design')
    parser.add_argument('--sc_num_denoising_steps', type=int, default=3,
                        help='Number of denoising steps for side chain packing')
    parser.add_argument('--sc_num_samples', type=int, default=16,
                        help='Number of samples for side chain packing')
    parser.add_argument('--repack_everything', action='store_true',
                        help='Repack all residues (not just designed ones)')
    # parser.add_argument('--pack_with_ligand_context', action='store_true', default=True,
    #                     help='Use ligand context during side chain packing')
    parser.add_argument('--force_hetatm', action='store_true',
                        help='Force ligand atoms to be written as HETATM')
    parser.add_argument('--zero_indexed', action='store_true',
                        help='Use zero-indexed residue numbering')
    
    # Filtering and selection parameters - simplified to single metric
    parser.add_argument('--selection_metric', type=str, default='ddg', 
                        choices=['mpnn', 'ddg', 'totalscore', 'res_totalscore', 'cms', 'ddg_after_relax_cst'],
                        help='Metric to use for selecting best structure')
    parser.add_argument('--selection_order', type=str, default='ascending', 
                        choices=['ascending', 'descending'],
                        help='Order for selection: ascending (lower is better) or descending (higher is better)')
    
    # Other settings
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save detailed statistics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--hb_atoms', type=str, default='',
                       help='Target ligand atom names for hydrogen bond calculation (comma-separated, e.g., "O1,O2,O3")')
    
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = parse_arguments()
    
    # Map the --use_side_chain_context flag to the ligand_mpnn_use_side_chain_context attribute
    if hasattr(args, 'use_side_chain_context') and args.use_side_chain_context:
        args.ligand_mpnn_use_side_chain_context = 1
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    try:
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)
        mpnn_fr = LigandMPNNFastRelax(args)
        cycle_results = mpnn_fr.run_iterative_design()
        
        # Get final structure from last cycle
        final_structure = cycle_results[-1]['relaxed_pdb'] if cycle_results else None
        
        print("\nIterative design completed successfully!")
        if final_structure:
            print(f"Final structure: {final_structure}")
            
            # Print summary of all cycles
            print(f"\nSummary of {len(cycle_results)} design cycles:")
            for result in cycle_results:
                cycle = result['cycle']
                seq_info = result['best_sequence']
                if 'final' in seq_info:
                    print(f"  Cycle {cycle}: Final score = {seq_info['final']:.4f}")
                    if 'mpnn' in seq_info and 'rosetta' in seq_info:
                        print(f"    MPNN: {seq_info['mpnn']:.4f}, Rosetta: {seq_info['rosetta']:.2f}")
                else:
                    print(f"  Cycle {cycle}: Score = {seq_info.get('score', 'N/A')}")
        else:
            print("No cycles completed successfully.")
        
    except Exception as e:
        print(f"Error during iterative design: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0
