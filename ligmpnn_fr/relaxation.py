"""Parallel FastRelax orchestration used by the pipeline."""

import multiprocessing as mp
import os
import queue
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from .relax_utils import fast_relax_worker, get_worker_config


def _worker_process(input_data, result_queue):
    result_queue.put(fast_relax_worker(input_data))


def _run_worker_with_timeout(input_data, timeout):
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    process = ctx.Process(target=_worker_process, args=(input_data, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            'success': False,
            'input_path': input_data[0],
            'output_path': input_data[1],
            'error': f"relaxation timed out after {timeout} seconds",
            'metrics': {},
        }
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            'success': False,
            'input_path': input_data[0],
            'output_path': input_data[1],
            'error': f"relaxation worker exited with code {process.exitcode} without a result",
            'metrics': {},
        }


class RelaxationMixin:
    def fast_relax_parallel(self, structure_paths):
        worker_config = get_worker_config(len(structure_paths), self.args)
        num_processes = worker_config['num_processes']
        pyrosetta_threads = worker_config['pyrosetta_threads']
        if self.args.verbose:
            print(f"Fast relax: {len(structure_paths)} structures, {num_processes} processes, {pyrosetta_threads} threads/process")
        
        # Get repackable residues and target atoms for constraints
        repackable_res = getattr(self.args, 'repackable_res', '')
        target_atm_for_cst = getattr(self.args, 'target_atm_for_cst', '')
        hb_atoms = getattr(self.args, 'hb_atoms', '')
        relax_mode = getattr(self.args, 'relax_mode', 'fastrelax')
        relax_timeout = getattr(self.args, 'relax_timeout', 0)
        
        worker_inputs = []
        for pdb_path in structure_paths:
            relaxed_path = pdb_path.replace('/backbones/', '/relaxed/').replace('.pdb', '_relaxed.pdb')
            os.makedirs(os.path.dirname(relaxed_path), exist_ok=True)
            worker_inputs.append((
                pdb_path, relaxed_path,
                self.args.ligand_params_path,
                self.args.use_genpot,
                self.args.verbose,
                pyrosetta_threads,
                repackable_res,
                target_atm_for_cst,
                hb_atoms,
                relax_mode
            ))

        successful_paths = []
        all_metrics = []
        failed_count = 0
        try:
            if relax_timeout and relax_timeout > 0:
                if self.args.verbose:
                    print(f"Relax timeout enabled: {relax_timeout} seconds per structure")
                for inp in worker_inputs:
                    result = _run_worker_with_timeout(inp, relax_timeout)
                    if result['success']:
                        successful_paths.append(result['output_path'])
                        all_metrics.append(result['metrics'])
                    else:
                        failed_count += 1
                        if self.args.verbose:
                            print(f"Failed {result['input_path']}: {result['error']}")
                        shutil.copy2(result['input_path'], result['output_path'])
                        successful_paths.append(result['output_path'])
                        all_metrics.append({})
            else:
                ctx = mp.get_context('spawn')
                with ProcessPoolExecutor(max_workers=num_processes, mp_context=ctx) as executor:
                    futures = {executor.submit(fast_relax_worker, inp): inp for inp in worker_inputs}
                    for future in as_completed(futures):
                        result = future.result()
                        if result['success']:
                            successful_paths.append(result['output_path'])
                            all_metrics.append(result['metrics'])
                        else:
                            failed_count += 1
                            if self.args.verbose:
                                print(f"Failed {result['input_path']}: {result['error']}")
                            shutil.copy2(result['input_path'], result['output_path'])
                            successful_paths.append(result['output_path'])
                            all_metrics.append({})  # Empty metrics for failed relaxation
            if self.args.verbose and failed_count > 0:
                print(f"Fast relax completed: {len(successful_paths)}/{len(structure_paths)} successful")
        except Exception as e:
            if self.args.verbose:
                print(f"Error during fast relax: {e}")
            raise e
        return successful_paths, all_metrics
