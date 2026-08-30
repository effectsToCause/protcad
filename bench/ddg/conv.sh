#!/bin/bash
# Is the seed spread convergence-limited or multi-basin?
# If sd(E_wt) falls as the sweep budget grows, the runs are simply not
# converged and the noise is a budget problem.  If it plateaus, each seed is
# landing in a genuinely different minimum and no amount of budget fixes it.
set -u
export CUDA_VISIBLE_DEVICES=1
BIN=$HOME/protcad/bin
PROT=${1:-1crn}
: > conv_${PROT}.txt
for sw in 2000 8000 32000 128000; do
  for s in $(seq 1 8); do
    PROTCAD_MC_SEED=$s $BIN/protMinRep $sw 1 $PROT.pdb conv_${sw}_$s.pdb 2>/dev/null \
      | awk -v w=$sw -v s=$s '/Ending Energy/{print w, s, $(NF-1)}' >> conv_${PROT}.txt
  done
done
echo done
