#!/usr/bin/env bash
# One full experiment round: nulls, baselines, L2, L3 (one task per GPU), then reporting.
# Everything below is a property of the configuration, not of this script: the loss comes
# from the target's type, the AR feature from the task's reference, the headline from
# whichever null is hardest in that year. This file only decides what runs where.
#
#   scripts/campaign.sh            # run it
#   scripts/campaign.sh --dry-run  # print the plan and the preconditions, run nothing

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=${LOG_DIR:-${EUMON_ARTEFACTS:-artefacts}/campaign_logs}
DRY=${1:-}

mkdir -p "$LOG"

plan() {
  cat <<'PLAN'
STAGE 1  gates            CPU     nulls scored before any model
STAGE 2  baselines        GPU 1   glm, glm_strata, nbgam, randomforest, convlstm
STAGE 3  l2               GPU 1   frozen probe, bfm + aurora
STAGE 4a l3 task A        GPU 1   LoRA r=4 and full FT, bfm + aurora, 500 epochs
STAGE 4b l3 task B        GPU 2   LoRA r=4 and full FT, bfm + aurora, 500 epochs
STAGE 4c l3 task C        GPU 3   LoRA r=4 and full FT, bfm + aurora, 500 epochs
STAGE 5  rescore..figures CPU     rescore, compute, tables, figures, provenance

The PEFT method ablation (lora1, lora16, vera, vera_asshipped) is run separately:
  scripts/run.py l3 --tasks A --ablation
PLAN
}

preconditions() {
  echo "--- preconditions ---"
  $PY scripts/audit.py >/dev/null 2>&1 && echo "  audit: PASS" || { echo "  audit: FAIL — stop"; return 1; }
  for g in 1 2 3; do
    $PY - "$g" <<'EOF'
import sys
from bfm_finetune.eumon.common.resources import gpu_occupants
g = int(sys.argv[1]); occ = gpu_occupants(g)
print(f"  GPU {g}: {'free' if not occ else 'BUSY ' + str(occ)}")
EOF
  done
  df -h . | tail -1 | awk '{print "  disk: " $4 " available"}'
}

plan
preconditions
if [ "$DRY" = "--dry-run" ]; then echo; echo "dry run — nothing started"; exit 0; fi

echo; echo "=== STAGE 1: gates (CPU) ==="
$PY scripts/run.py gates --run-id campaign 2>&1 | tee "$LOG/gates.log"

echo; echo "=== STAGE 2: baselines (GPU 1) ==="
$PY scripts/run.py baselines --gpu 1 --seeds 1 --run-id campaign 2>&1 | tee "$LOG/baselines.log"

echo; echo "=== STAGE 3: L2 frozen probe, both backbones (GPU 1) ==="
$PY scripts/run.py l2 --gpu 1 --backbones bfm aurora --seeds 1 \
    --run-id campaign 2>&1 | tee "$LOG/l2.log"

echo; echo "=== STAGE 4: L3 LoRA r=4 + full FT, one task per GPU, both backbones ==="
$PY scripts/run.py l3 --gpu 1 --backbones bfm aurora --base-arms lora4 full \
    --tasks A --seeds 1 --l3-epochs 500 --l3-patience 50 --run-id campaign_l3_A \
    > "$LOG/l3_A.log" 2>&1 &
P1=$!
$PY scripts/run.py l3 --gpu 2 --backbones bfm aurora --base-arms lora4 full \
    --tasks B --seeds 1 --l3-epochs 500 --l3-patience 50 --run-id campaign_l3_B \
    > "$LOG/l3_B.log" 2>&1 &
P2=$!
$PY scripts/run.py l3 --gpu 3 --backbones bfm aurora --base-arms lora4 full \
    --tasks C --seeds 1 --l3-epochs 500 --l3-patience 50 --run-id campaign_l3_C \
    > "$LOG/l3_C.log" 2>&1 &
P3=$!
echo "  task A=$P1 (GPU 1)  task B=$P2 (GPU 2)  task C=$P3 (GPU 3)"
FAILED=0
for p in $P1 $P2 $P3; do wait $p || FAILED=1; done
[ $FAILED -eq 0 ] || { echo "a stream failed — see $LOG"; exit 1; }

echo; echo "=== STAGE 5: score and report (CPU) ==="
$PY scripts/run.py rescore compute tables figures provenance --run-id campaign 2>&1 \
    | tee "$LOG/report.log"

echo; echo "campaign complete — tables and costs under the artefacts root"
