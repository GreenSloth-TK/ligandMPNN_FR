"""Structure selection helpers."""


class SelectionMixin:
    def _select_best_relaxed_structure(self, sequences, relaxed_paths, all_metrics, cycle_num):
        """
        Select best relaxed structure based on single metric (simplified from combined scoring)
        """
        if self.args.verbose:
            print(f"Evaluating {len(sequences)} relaxed structures to select best...")
            print(f"Using selection metric: {self.args.selection_metric} ({self.args.selection_order})")
            
        best_score = float('inf') if self.args.selection_order == 'ascending' else float('-inf')
        best_sequence = None
        best_path = None
        
        # Store individual scores for statistics
        all_scores = []
        
        # Extract scores from sequences and metrics
        mpnn_scores = [seq_data['score'] for seq_data in sequences]
        
        if self.args.verbose:
            print(f"MPNN scores range: {min(mpnn_scores):.4f} to {max(mpnn_scores):.4f}")
            
        # Extract all available metrics for display
        metric_values = {}
        for metric in ['ddg', 'totalscore', 'res_totalscore', 'cms', 'ddg_after_relax_cst']:
            values = [metrics.get(metric, float('inf')) for metrics in all_metrics]
            metric_values[metric] = values
            if any(v != float('inf') for v in values):
                print(f"{metric.upper()} range: {min(v for v in values if v != float('inf')):.3f} to {max(v for v in values if v != float('inf')):.3f}")
        
        # Select best structure based on chosen metric
        for i, (seq_data, relaxed_path) in enumerate(zip(sequences, relaxed_paths)):
            mpnn_score = mpnn_scores[i]
            structure_metrics = all_metrics[i]
            
            # Get the value for the selection metric
            if self.args.selection_metric == 'mpnn':
                selection_value = mpnn_score
            else:
                selection_value = structure_metrics.get(self.args.selection_metric, float('inf'))
            
            # Store all scores for this structure
            score_components = {
                'mpnn': mpnn_score,
                'selection_metric': self.args.selection_metric,
                'selection_value': selection_value,
                'metrics': structure_metrics
            }
            
            all_scores.append(score_components)
            
            # Track best structure based on selection order
            is_better = False
            if self.args.selection_order == 'ascending':
                is_better = selection_value < best_score
            else:  # descending
                is_better = selection_value > best_score
                
            if is_better:
                best_score = selection_value
                best_sequence = seq_data.copy()
                best_sequence.update(score_components)
                best_path = relaxed_path
                
            if self.args.verbose:
                print(f"Structure {i}: MPNN={mpnn_score:.4f}, {self.args.selection_metric.upper()}={selection_value:.3f}")
                if structure_metrics:
                    # Enhanced metric display with all available values
                    metric_items = []
                    for k, v in structure_metrics.items():
                        if isinstance(v, (int, float)):
                            if k in ['totalscore', 'ddg', 'ddg_after_relax_cst']:
                                metric_items.append(f"{k}={v:.2f}")
                            elif k == 'cms':
                                metric_items.append(f"{k}={v:.4f}")
                            elif k == 'res_totalscore':
                                metric_items.append(f"{k}={v:.3f}")
                            else:
                                metric_items.append(f"{k}={v:.2f}")
                    if metric_items:
                        print(f"  All metrics: {', '.join(metric_items)}")
        
        # Handle case where no valid structures found
        if best_sequence is None:
            print("ERROR: All fast relax jobs failed! No valid structures to select from.")
            if sequences:
                best_sequence = sequences[0].copy()
                best_sequence.update({
                    'selection_metric': self.args.selection_metric,
                    'selection_value': float('inf'),
                    'metrics': {}
                })
                best_path = relaxed_paths[0] if relaxed_paths else None
            else:
                raise ValueError("No sequences available for selection")
        
        # Store all scores in best sequence for statistics
        best_sequence['all_scores'] = all_scores
        
        if self.args.verbose:
            print("\nBest structure selected:")
            print(f"  Selection metric ({self.args.selection_metric}): {best_score:.4f}")
            print(f"  MPNN score: {best_sequence.get('mpnn', 'N/A'):.4f}")
            if 'metrics' in best_sequence and best_sequence['metrics']:
                metrics = best_sequence['metrics']
                print(f"  DDG: {metrics.get('ddg', 'N/A'):.2f}")
                print(f"  Total score: {metrics.get('totalscore', 'N/A'):.2f}")
                print(f"  Contact surface: {metrics.get('cms', 'N/A'):.4f}")
                
        return best_sequence, best_path
