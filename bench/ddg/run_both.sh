#!/bin/bash
set -u
cd "$(dirname "$0")"
for LG in 0 1; do
  echo "=== PROTCAD_ENERGY_LEGACY=$LG ==="
  PROTCAD_ENERGY_LEGACY=$LG ./ddg_cancel.sh 1crn 36000 4 32
  mv ddg_1crn.txt      ddg_1crn_lg$LG.txt
  mv ddg_1crn_full.txt ddg_1crn_full_lg$LG.txt
done
echo ALLDONE
