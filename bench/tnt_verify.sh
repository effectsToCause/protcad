#!/bin/bash
# Byte-identical oracle for the TNT removal.
#
# Vec3/Mat3 preserve TNT's arithmetic order, so agreement should be exact at
# PDB precision.  It is NOT exact to the last ULP: returning Vec3 by value
# lets the compiler keep values in registers and contract multiply-adds into
# FMA, where TNT's heap storage forced stores and loads.  That shows up only
# in code that argmins over mathematically degenerate values -- see the
# terminusTest note in RESULTS.
#
# protMin is deliberately excluded.  It is nondeterministic run to run
# independently of this change -- the pre-removal build produced different
# output for two identical invocations -- so it cannot serve as an oracle.
set -u
export CUDA_VISIBLE_DEVICES=1   # P2200: correctness only, never training
TAG=${1:?usage: tnt_verify.sh <tag> [bindir]}
BIN=${2:-$HOME/protcad/bin}
OUT=/tmp/tntver_$TAG; rm -rf $OUT; mkdir -p $OUT
PDB=$HOME/protcad/tests/data/1crn.pdb
strip() { grep -viE 'elapsed|took|wall|per-candidate|time|clock|GPU [0-9]|device'; }

for t in amberParamsTest batchDeltaTest batchTest bestfitTest clashCheck \
         energyTerms energyTest frozenDielectricTest fusionTest \
         invarianceTest replicaTest rotamerTest terminusTest; do
  $BIN/$t $PDB > $OUT/$t.raw 2>&1
  strip < $OUT/$t.raw > $OUT/$t.txt
done

for s in 1 2; do
  PROTCAD_MC_SEED=$s PROTCAD_MC_BACKBONE=0 \
    $BIN/protMinRep 3000 32 $PDB $OUT/mr_$s.pdb > $OUT/mr_$s.raw 2>&1
  strip < $OUT/mr_$s.raw > $OUT/mr_$s.txt
done

( cd $OUT && md5sum *.txt *.pdb 2>/dev/null | sort -k2 ) > $OUT/MANIFEST
echo "manifest -> $OUT/MANIFEST ($(wc -l < $OUT/MANIFEST) entries)"
