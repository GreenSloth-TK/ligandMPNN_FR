"""Hydrogen-bond metric helpers."""

from collections import defaultdict


def calc_hb(pose, nres, target_hb_atms):
    import pyrosetta

    if isinstance(target_hb_atms, str):
        target_hb_atms = target_hb_atms.split(',')
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pose.update_residue_neighbors()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    #
    lig_res = pose.residue(nres)
    lig_atm_hb = defaultdict(list)
    for lig_atmName in target_hb_atms:
        atm_idx = lig_res.atom_index(lig_atmName)
        atm_id = pyrosetta.rosetta.core.id.AtomID(atm_idx, nres)
        found_hbs = hbond_set.atom_hbonds(atm_id)
        #
        if (len(found_hbs) == 0):
            continue
        for hb in found_hbs:
            don_resNo = hb.don_res()
            don_atmName = pose.residue(don_resNo).atom_name(hb.don_hatm())
            acc_resNo = hb.acc_res()
            acc_atmName = pose.residue(acc_resNo).atom_name(hb.acc_atm())
            #
            hb_atm = {don_resNo: don_atmName, acc_resNo: acc_atmName}
            #
            hb_res_pair = [don_resNo, acc_resNo]
            for i_res, resno in enumerate(hb_res_pair):
                if resno == nres:
                    other_resno = hb_res_pair[1 - i_res]
                    if other_resno == resno:
                        continue
                    lig_atm_hb[hb_atm[resno].strip()].append((other_resno, hb_atm[other_resno]))
            hb_sc = {}
    for lig_atmName in target_hb_atms:
        if lig_atmName not in list(lig_atm_hb.keys()):
            hb_sc['%s_hbond'%lig_atmName] = 0
        else:
            tmp = []
            for hb in lig_atm_hb[lig_atmName]:
                if hb not in tmp:
                    tmp.append(hb)
            hb_sc['%s_hbond'%lig_atmName] = len(tmp)
    return hb_sc
