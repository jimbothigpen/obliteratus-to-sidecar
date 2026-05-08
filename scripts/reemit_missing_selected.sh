#!/bin/bash
# Re-run any (model, method) pair that has a `*.abl.gguf` but no
# matching `*.selected.abl.gguf`. Drops the existing full sidecar so the
# matrix driver's resume-logic re-runs the extraction with the dual-emit
# adapter.
#
# Use AFTER the main matrix completes — running this concurrently with the
# matrix would race for the GPU.

set -uo pipefail

CEPH=/mnt/cephfs/0/Container/systems/ai00/users/builduser
SIDECAR_DIR=$CEPH/sidecars

deleted=0
for f in "$SIDECAR_DIR"/*/*.abl.gguf; do
    # skip the .selected.abl.gguf siblings
    case "$f" in
        *.selected.abl.gguf) continue ;;
    esac
    base=${f%.abl.gguf}
    if [[ ! -f "${base}.selected.abl.gguf" ]]; then
        echo "MISSING-SELECTED: $f → deleting full so matrix re-runs it"
        rm -f "$f" "${base}.abl.meta.json"
        deleted=$((deleted+1))
    fi
done

echo "Deleted $deleted full sidecars without selected siblings."
echo "Now re-run scripts/run_validation_matrix.sh to re-emit them with dual variants."
