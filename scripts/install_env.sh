#!/usr/bin/env bash
# Create the ligmpnn-fr conda environment from environment.yaml and apply
# a numpy compatibility fix for LigandMPNN's bundled openfold library.
# openfold uses removed aliases (np.int, np.bool, etc.) that were dropped
# in NumPy 1.24+. Python 3.12 requires NumPy >= 1.26, so downgrading is not
# possible — we patch at the environment level via sitecustomize.py instead.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="ligmpnn-fr"
ENV_FILE="$REPO_DIR/environment.yaml"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' from $ENV_FILE ..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

SITE_PACKAGES="$(conda run -n "$ENV_NAME" python -c 'import site; print(site.getsitepackages()[0])')"
SITECUSTOMIZE="$SITE_PACKAGES/sitecustomize.py"

echo "Installing numpy compatibility shim -> $SITECUSTOMIZE"
cat > "$SITECUSTOMIZE" <<'PYEOF'
try:
    import warnings
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'complex'):
            np.complex = complex
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'object'):
            np.object = object
        if not hasattr(np, 'str'):
            np.str = str
except ImportError:
    pass
PYEOF

echo "Verifying fix..."
conda run -n "$ENV_NAME" python -c "import numpy as np; np.int; np.bool; print('numpy aliases OK')"
echo "Done. Activate with: conda activate $ENV_NAME"
