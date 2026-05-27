#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  LMPNN_DIR=/path/to/LigandMPNN PDB_PATH=/path/to/complex.pdb LIGAND_PARAMS_PATH=/path/to/ligand.params scripts/test_run.sh

Required environment variables:
  LMPNN_DIR              LigandMPNN repository path

Optional environment variables:
  MODEL_WEIGHTS_DIR      LigandMPNN model weights path (default: $LMPNN_DIR/model_params)
  OUT_DIR                Output directory (default: test_output/ligmpnn_fr)
  N_CYCLES               Number of design/relax cycles (default: 3)
  NUM_SEQ_PER_TARGET     Number of sequences per cycle (default: 4)
  NUM_PROCESSES          FastRelax processes (default: 1)
  PYROSETTA_THREADS      PyRosetta threads per process (default: 1)
  RELAX_MODE             fastrelax or score_only (default: score_only)
  RELAX_TIMEOUT          Seconds before worker timeout; 0 disables (default: 300)
  TEMPERATURE            LigandMPNN sampling temperature (default: 0.1)
  PACK_SIDE_CHAINS       Enable LigandMPNN side chain packing; set to 1 to enable (default: off)
  CHECKPOINT_PATH_SC     Side chain packer checkpoint path (default: $LMPNN_DIR/ligandmpnn_sc_v_32_002_16.pt)
  TARGET_ATM_FOR_CST     Ligand atom names for distance constraints, comma-separated (e.g. "O1,O2,O3")
  HB_ATOMS               Ligand atom names for H-bond counting, comma-separated (e.g. "O1,O2,O3")
  OMIT_AA                Global AA omit, e.g. "CP" (default: X)
  BIAS_AA                Global AA bias, e.g. "A:-1.0,P:2.3" (default: off)
  BIAS_AA_PER_RESIDUE    Path to JSON for per-residue AA bias, e.g. {"A12": {"G": -0.3}} (default: off)
  OMIT_AA_PER_RESIDUE    Path to JSON for per-residue AA omit, e.g. {"A12": "PG"} (default: off)
  EXTRA_ARGS             Additional arguments passed to ligmpnn_fr
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LMPNN_DIR:-}" ]]; then
    echo "ERROR: LMPNN_DIR is not set." >&2
    echo "  export LMPNN_DIR=/path/to/LigandMPNN" >&2
    exit 2
fi
: "${MODEL_WEIGHTS_DIR:=$LMPNN_DIR/model_params}"
: "${OUT_DIR:=$repo_dir/test_output/ligmpnn_fr}"
: "${N_CYCLES:=2}"
: "${NUM_SEQ_PER_TARGET:=4}"
: "${NUM_PROCESSES:=1}"
: "${PYROSETTA_THREADS:=1}"
: "${RELAX_MODE:=score_only}"
: "${RELAX_TIMEOUT:=300}"
: "${TEMPERATURE:=0.1}"
: "${PACK_SIDE_CHAINS:=}"
: "${CHECKPOINT_PATH_SC:=}"
: "${TARGET_ATM_FOR_CST:=}"
: "${HB_ATOMS:=}"
: "${OMIT_AA:=}"
: "${BIAS_AA:=}"
: "${BIAS_AA_PER_RESIDUE:=}"
: "${OMIT_AA_PER_RESIDUE:=}"
: "${EXTRA_ARGS:=}"

if [[ -z "${PDB_PATH:-}" || -z "${LIGAND_PARAMS_PATH:-}" ]]; then
    usage >&2
    echo >&2
    echo "ERROR: PDB_PATH and LIGAND_PARAMS_PATH are required." >&2
    exit 2
fi

if [[ ! -f "$PDB_PATH" ]]; then
    echo "ERROR: PDB_PATH does not exist: $PDB_PATH" >&2
    exit 2
fi

if [[ ! -f "$LIGAND_PARAMS_PATH" ]]; then
    echo "ERROR: LIGAND_PARAMS_PATH does not exist: $LIGAND_PARAMS_PATH" >&2
    exit 2
fi

if [[ ! -d "$LMPNN_DIR" ]]; then
    echo "ERROR: LMPNN_DIR does not exist: $LMPNN_DIR" >&2
    exit 2
fi

export LMPNN_DIR
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="$LMPNN_DIR:$repo_dir:${PYTHONPATH:-}"
export ROSETTA_NUM_THREADS="$PYROSETTA_THREADS"

mkdir -p "$OUT_DIR"

optional_args=()
[[ "${PACK_SIDE_CHAINS}" == "1" ]] && optional_args+=(--pack_side_chains)
[[ -n "${CHECKPOINT_PATH_SC}" ]]   && optional_args+=(--checkpoint_path_sc "$CHECKPOINT_PATH_SC")
[[ -n "${TARGET_ATM_FOR_CST}" ]]   && optional_args+=(--target_atm_for_cst "$TARGET_ATM_FOR_CST")
[[ -n "${HB_ATOMS}" ]]             && optional_args+=(--hb_atoms "$HB_ATOMS")
[[ -n "${OMIT_AA}" ]]              && optional_args+=(--omit_AA "$OMIT_AA")
[[ -n "${BIAS_AA}" ]]              && optional_args+=(--bias_AA "$BIAS_AA")
[[ -n "${BIAS_AA_PER_RESIDUE}" ]]  && optional_args+=(--bias_AA_per_residue "$BIAS_AA_PER_RESIDUE")
[[ -n "${OMIT_AA_PER_RESIDUE}" ]]  && optional_args+=(--omit_AA_per_residue "$OMIT_AA_PER_RESIDUE")

python -m ligmpnn_fr \
    --pdb_path "$PDB_PATH" \
    --ligand_params_path "$LIGAND_PARAMS_PATH" \
    --path_to_model_weights "$MODEL_WEIGHTS_DIR" \
    --out_folder "$OUT_DIR" \
    --n_cycles "$N_CYCLES" \
    --num_seq_per_target "$NUM_SEQ_PER_TARGET" \
    --num_processes "$NUM_PROCESSES" \
    --pyrosetta_threads "$PYROSETTA_THREADS" \
    --relax_mode "$RELAX_MODE" \
    --relax_timeout "$RELAX_TIMEOUT" \
    --temperature "$TEMPERATURE" \
    --save_stats \
    "${optional_args[@]}" \
    $EXTRA_ARGS
