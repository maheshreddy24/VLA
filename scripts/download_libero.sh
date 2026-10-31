#!/bin/bash
# Downloads the LIBERO demonstrations (HDF5) to data/libero/<suite>/.
#   bash scripts/download_libero.sh                        # spatial, object, goal, 10
#   bash scripts/download_libero.sh libero_spatial         # one suite
set -e
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(libero_spatial libero_object libero_goal libero_10)
for s in "${SUITES[@]}"; do
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', allow_patterns='${s}/*', local_dir='${ROOT}/data/libero')
"
    echo "${s}: $(ls "${ROOT}/data/libero/${s}" | wc -l) files"
done
