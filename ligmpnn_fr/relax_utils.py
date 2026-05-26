"""PyRosetta FastRelax worker utilities."""

import os
import tempfile
from multiprocessing import cpu_count

from .gen_prot_lig_dist_cst import extract_dist_cst_from_pdb
from .hbonds import calc_hb
from .xml_relax_after_ligMPNN import XML_BSITE_FASTRELAX


def fast_relax_worker(input_data):
    """
    Worker function for multiprocessing fast relax with simplified XML-based protocol
    Each process gets its own PyRosetta instance and uses minimal threading
    Extracts key metrics: ddg, cms, totalscore
    """
    if len(input_data) == 10:
        pdb_path, output_path, ligand_params_path, use_genpot, verbose, pyrosetta_threads, repackable_res, target_atm_for_cst, hb_atoms, relax_mode = input_data
    else:
        pdb_path, output_path, ligand_params_path, use_genpot, verbose, pyrosetta_threads, repackable_res, target_atm_for_cst, hb_atoms = input_data
        relax_mode = 'fastrelax'
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

        if verbose:
            print(f"[relax-worker] init PyRosetta for {pdb_path}", flush=True)

        # Improved init flags - use flexible threading
        init_flags = ['-beta', '-mute all']  # Always mute to reduce log spam
        
        # Use configured number of threads per process (now allowing more than 1)
        init_flags.extend([
            f'-multithreading:total_threads {pyrosetta_threads}',
            f'-multithreading:interaction_graph_threads {pyrosetta_threads}'
        ])
        
        # Add ligand parameters
        if use_genpot:
            init_flags.extend(['-gen_potential', f'-extra_res_fa {ligand_params_path}'])
        else:
            init_flags.append(f'-extra_res_fa {ligand_params_path}')
            
        # Add native structure for coordinate constraints
        init_flags.append(f'-in:file:native {pdb_path}')

        # Initialize PyRosetta
        pyrosetta.init(' '.join(init_flags))

        # Load pose
        if verbose:
            print(f"[relax-worker] loading pose: {pdb_path}", flush=True)
        pose = pyrosetta.pose_from_pdb(pdb_path)
        if pose.total_residue() == 0:
            raise ValueError("Empty pose loaded")

        if relax_mode == 'score_only':
            if verbose:
                print("[relax-worker] score_only mode: skipping RosettaScripts FastRelax", flush=True)
            from pyrosetta.rosetta.core.scoring import get_score_function
            scorefxn = get_score_function()
            total_energy = scorefxn(pose)
            metrics = {
                'totalscore': total_energy,
                'ddg': total_energy,
                'cms': 0.0,
                'res_totalscore': total_energy / pose.total_residue(),
                'ddg_after_relax_cst': total_energy,
            }
            hb_d = {'total_hb': 0}
            if hb_atoms:
                try:
                    hb_d = calc_hb(pose, len(pose), hb_atoms)
                    hb_d['total_hb'] = sum(hb_d.values())
                except Exception:
                    hb_d = {'total_hb': 0}
            metrics.update(hb_d)
            pose.dump_pdb(output_path)
            return {
                'success': True,
                'input_path': pdb_path,
                'output_path': output_path,
                'error': None,
                'metrics': metrics,
            }

        # Generate distance constraints if target atoms specified
        cst_file_path = None
        if target_atm_for_cst:
            try:
                target_atoms = target_atm_for_cst.split(',') if isinstance(target_atm_for_cst, str) else target_atm_for_cst
                if target_atoms and target_atoms != ['']:
                    # Generate constraints
                    if verbose:
                        print(f"Generating constraints for atoms: {target_atoms}")
                    csts = extract_dist_cst_from_pdb(pdb_path, target_atoms, bsite_res=repackable_res)
                    if csts:
                        # Create temporary constraint file
                        cst_fd, cst_file_path = tempfile.mkstemp(suffix='.cst', text=True)
                        with os.fdopen(cst_fd, 'w') as f:
                            f.write('\n'.join(csts) + '\n')
                        if verbose:
                            print(f"Created constraint file with {len(csts)} constraints")
                    else:
                        if verbose:
                            print("No constraints generated")
                else:
                    if verbose:
                        print("Empty target atoms, skipping constraint generation")
            except Exception as e:
                if verbose:
                    print(f"Error generating constraints: {e}")
                    print(f"target_atm_for_cst: {repr(target_atm_for_cst)}")
                    print(f"target_atoms: {repr(target_atoms) if 'target_atoms' in locals() else 'not defined'}")
                # Continue with empty constraint file
                pass

        # If no constraint file, create empty temp file
        if cst_file_path is None:
            cst_fd, cst_file_path = tempfile.mkstemp(suffix='.cst', text=True)
            with os.fdopen(cst_fd, 'w') as f:
                f.write('')

        # Prepare simplified XML script with repackable residues
        if not repackable_res:
            # If no repackable residues specified, get all protein residues
            repack_res_str = ','.join([str(i) for i in range(1, pose.total_residue())])
        else:
            repack_res_str = repackable_res

        # Use the complete XML template with full DDG calculations
        complete_xml = XML_BSITE_FASTRELAX.format(repack_res_str, cst_file_path)
        
        # Create temporary XML file
        xml_fd, xml_file_path = tempfile.mkstemp(suffix='.xml', text=True)
        with os.fdopen(xml_fd, 'w') as f:
            f.write(complete_xml)

        metrics = {}
        try:
            # Parse and run XML script
            if verbose:
                print("[relax-worker] parsing RosettaScripts XML", flush=True)
            objs = XmlObjects.create_from_file(xml_file_path)
            
            # Apply the protocol
            protocol = objs.get_mover('ParsedProtocol')
            if verbose:
                print("[relax-worker] applying RosettaScripts protocol", flush=True)
            protocol.apply(pose)
            if verbose:
                print("[relax-worker] RosettaScripts protocol finished", flush=True)
            #apply 이후, pose는 relaxed 상태.

            # Extract comprehensive metrics
            try:
                # Get contact molecular surface
                cms_filter = objs.get_filter('cms') 
                if cms_filter:
                    metrics['cms'] = cms_filter.report_sm(pose)
                
                # Get total score
                totalscore_filter = objs.get_filter('totalscore')
                if totalscore_filter:
                    metrics['totalscore'] = totalscore_filter.report_sm(pose)
                
                # Get residue-normalized total score
                res_totalscore_filter = objs.get_filter('res_totalscore')
                if res_totalscore_filter:
                    metrics['res_totalscore'] = res_totalscore_filter.report_sm(pose)
                
                # Get DDG after relaxation with constraints
                ddg_after_relax_filter = objs.get_filter('ddg_after_relax_cst')
                if ddg_after_relax_filter:
                    metrics['ddg_after_relax_cst'] = ddg_after_relax_filter.report_sm(pose)
                
                # Get DDG without constraints (more accurate binding energy)
                ddg_filter = objs.get_filter('ddg')
                if ddg_filter:
                    metrics['ddg'] = ddg_filter.report_sm(pose)
                else:
                    # Fallback to DDG with constraints
                    metrics['ddg'] = metrics.get('ddg_after_relax_cst', metrics.get('totalscore', 0.0))

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract some metrics: {e}")
                # Set fallback values for missing metrics
                from pyrosetta.rosetta.core.scoring import get_score_function
                scorefxn = get_score_function()
                total_energy = scorefxn(pose)
                metrics['totalscore'] = total_energy
                metrics['ddg'] = total_energy  # Use total score as ddg fallback
                metrics['cms'] = 0.0
                metrics['res_totalscore'] = total_energy / pose.total_residue()
                metrics['ddg_after_relax_cst'] = total_energy
           
            # Get hydrogen bond metrics
            hb_d = {'total_hb': 0}
            if hb_atoms:
                try:
                    hb_d = calc_hb(pose, len(pose), hb_atoms)
                    total_hb = sum(hb_d.values())
                    hb_d['total_hb'] = total_hb

                except Exception as e:
                    hb_d = {'total_hb': 0}
            metrics.update(hb_d)
                    
        finally:
            # Clean up temporary files
            try:
                os.unlink(xml_file_path)
                os.unlink(cst_file_path)
            except Exception:
                pass

        # Save output
        pose.dump_pdb(output_path)
        if not os.path.exists(output_path):
            raise RuntimeError("Output PDB file was not created")

        return {
            'success': True, 
            'input_path': pdb_path, 
            'output_path': output_path, 
            'error': None,
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'success': False, 
            'input_path': pdb_path, 
            'output_path': output_path, 
            'error': str(e),
            'metrics': {}
        }


def get_worker_config(num_structures, args):
    available_cores = cpu_count()
    num_processes = min(
        getattr(args, 'num_processes', 1),
        num_structures,
        available_cores
    )
    pyrosetta_threads = getattr(args, 'pyrosetta_threads', 2)
    total_threads = num_processes * pyrosetta_threads
    if total_threads > available_cores:
        print(f"Warning: total threads ({total_threads}) exceed available CPU cores ({available_cores}). "
              "This may cause oversubscription and reduce performance.")
    return {
        'num_processes': num_processes,
        'pyrosetta_threads': pyrosetta_threads
    }
