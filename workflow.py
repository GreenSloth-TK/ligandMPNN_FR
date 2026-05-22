"""High-level iterative design workflow."""

import json
import os
import re
import shutil
import time


class WorkflowMixin:
    def _setup_output_dirs(self):
        """Create output directories"""
        self.base_folder = self.args.out_folder
        if self.base_folder[-1] != "/":
            self.base_folder += "/"
            
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(f"{self.base_folder}seqs", exist_ok=True)
        os.makedirs(f"{self.base_folder}backbones", exist_ok=True)
        os.makedirs(f"{self.base_folder}relaxed", exist_ok=True)
        
        if self.args.save_stats:
            os.makedirs(f"{self.base_folder}stats", exist_ok=True)

    def run_mpnn_fastrelax_cycle(self, pdb_path, cycle_num=1):
        """
        Run one cycle of LigandMPNN design + fast relax with parallel processing
        
        Args:
            pdb_path: Input PDB path
            cycle_num: Current cycle number
            
        Returns:
            Path to best relaxed PDB file
        """
        if self.args.verbose:
            print(f"\n{'='*50}")
            print(f"Starting cycle {cycle_num}")
            print(f"{'='*50}")
            
        # Generate sequences
        sequences = self.generate_sequences(
            pdb_path,
            num_sequences=self.args.num_seq_per_target,
            temperature=self.args.temperature,
            fixed_residues=getattr(self.args, 'fixed_residues', []),
            redesigned_residues=getattr(self.args, 'redesigned_residues', [])
        )
        
        if self.args.verbose:
            print(f"Generated {len(sequences)} sequences for cycle {cycle_num}")
            scores = [seq['score'] for seq in sequences]
            print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
        
        # Create structures for all sequences (if multiple sequences generated)
        header = os.path.basename(pdb_path).replace('.pdb', '').replace('.cif', '')
        header = re.sub(r'(_cycle_\d+_relaxed)+', '', header)

        # Multiple sequences - create structures for all and parallel relax
        cycle_tag = f"{header}_cycle_{cycle_num}"
        structure_paths = []
        for i, seq_data in enumerate(sequences):
            threaded_path = f"{self.base_folder}backbones/{cycle_tag}_seq_{i}.pdb"
            self.create_structure_with_sequence(
                pdb_path, 
                seq_data['sequence'], 
                seq_data['S_sample'], 
                threaded_path
            )
            structure_paths.append(threaded_path)
        
        # Perform fast relax (parallel if multiple structures)
        # Multiple structures - use parallel relaxation
        if self.args.verbose:
            print(f"Performing parallel fast relax on {len(structure_paths)} structures")
            
        relaxed_paths, all_metrics = self.fast_relax_parallel(
            structure_paths, 
            # max_workers=getattr(self.args, 'max_relax_workers', None)
        )
        
        # Evaluate relaxed structures and select best
        best_sequence, best_relaxed_path = self._select_best_relaxed_structure(
            sequences, relaxed_paths, all_metrics, cycle_num
        )
        
        # Move best structure to final location
        final_relaxed_path = f"{self.base_folder}relaxed/{cycle_tag}_relaxed.pdb"
        if best_relaxed_path != final_relaxed_path:
            shutil.copy2(best_relaxed_path, final_relaxed_path)
        relaxed_paths = [final_relaxed_path]
        
        # Save sequence information
        seq_path_best = f"{self.base_folder}seqs/{cycle_tag}_best.fa"
        with open(seq_path_best, 'w') as f:
            f.write(f">cycle_{cycle_num}_score_{best_sequence['score']:.4f}\n")
            f.write(f"{best_sequence['sequence']}\n")

        seq_path = f"{self.base_folder}seqs/{cycle_tag}.fa"
        with open(seq_path, 'w') as f:
            for i, seq_data in enumerate(sequences):
                f.write(f">{cycle_tag}_seq_{i}\n")
                f.write(f"{seq_data['sequence']}\n")

        if self.args.save_stats:
            stats_path = f"{self.base_folder}stats/{cycle_tag}.json"
            stats = {
                'cycle': cycle_num,
                'input_pdb': pdb_path,
                'relaxed_pdb': relaxed_paths[0],
                'best_sequence': best_sequence['sequence'],
                'best_score': best_sequence['score'],
                'temperature': best_sequence['temperature'],
                'num_sequences_generated': len(sequences),
                'sequence_diversity': len(set([seq['sequence'] for seq in sequences])),
                # Simplified selection method
                'selection_metric': self.args.selection_metric,
                'selection_order': self.args.selection_order,
                'selection_value': best_sequence.get('selection_value', None),
                # Individual scores for best structure
                'best_mpnn_score': best_sequence.get('mpnn', best_sequence['score']),
                # All available metrics for best structure
                'best_metrics': best_sequence.get('metrics', {}),
                # All scores for analysis
                'all_mpnn_scores': [seq['score'] for seq in sequences],
                'all_relax_scores': best_sequence.get('all_scores', []),
            }
                
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        if self.args.verbose:
            print(f"Cycle {cycle_num} completed")
            print(f"Best sequence: {best_sequence['sequence']}")
            print(f"Best MPNN score: {best_sequence.get('mpnn', best_sequence['score']):.4f}")
            print(f"Selection metric ({self.args.selection_metric}): {best_sequence.get('selection_value', 'N/A'):.4f}")
            
            # Show additional XML metrics if available
            if 'metrics' in best_sequence and best_sequence['metrics']:
                metrics = best_sequence['metrics']
                metric_items = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key in ['ddg', 'totalscore', 'ddg_after_relax_cst']:
                            metric_items.append(f"{key}={value:.2f}")
                        elif key == 'cms':
                            metric_items.append(f"{key}={value:.4f}")
                        elif key == 'res_totalscore':
                            metric_items.append(f"{key}={value:.3f}")
                        else:
                            metric_items.append(f"{key}={value:.2f}")
                if metric_items:
                    print(f"All metrics: {', '.join(metric_items)}")
            print(f"Final relaxed structure: {relaxed_paths[0]}")
            
        return best_sequence, relaxed_paths[0]

    def run_iterative_design(self):
        """Run iterative LigandMPNN + FastRelax cycles"""
        current_pdb = self.args.pdb_path
        all_cycle_results = []
        
        start_time = time.time()
        
        for cycle in range(1, self.args.n_cycles + 1):
            # Run MPNN + FastRelax cycle
            best_sequence, relaxed_pdb = self.run_mpnn_fastrelax_cycle(current_pdb, cycle)
            
            # Store cycle results
            all_cycle_results.append({
                'cycle': cycle,
                'best_sequence': best_sequence,
                'relaxed_pdb': relaxed_pdb
            })
            
            # Use relaxed structure as input for next cycle
            current_pdb = relaxed_pdb
                
        end_time = time.time()
        
        if self.args.verbose:
            print("\nIterative design completed!")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Final structure: {current_pdb}")
            
        return all_cycle_results
