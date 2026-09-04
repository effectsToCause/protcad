# ddG on 1crn: what is settled and what is not

Configuration: 36000 sweeps, 32 replicas, 4 seeds, folded state with a rigid
backbone, unfolded reference Gly-X-Gly with a mobile backbone, reported
quantity `A = <E> - T*S_conf` rather than the best conformation seen.

## The pre-registered parameter-set arm resolves

| arm | sd(E_wt) | sd(matched) | verdict |
|---|---|---|---|
| default (`PROTCAD_ENERGY_LEGACY=0`) | 0.28 | 0.33 - 0.67 | passes |
| legacy (`PROTCAD_ENERGY_LEGACY=1`)  | 2.93 | 8.47 - 9.83 | fails, > 1.5 |

Legacy is out. Everything below is the default set.

## Basin pairing turned out to be unnecessary rather than wrong

The harness exists because a single minimisation carried 5-6 kcal/mol of seed
noise against a 1-2 kcal/mol signal, and the hope was that pairing WT and
mutant into a shared basin would cancel it. Measured unpaired sd is now
0.33-0.37. The noise it was built to fight was an artefact of reporting the
best-seen conformation from one replica; an ensemble average over 32 replicas
removes it at the source. Reported cancellation percentages are meaningless
in this regime (they come out negative) because the denominator is already at
the noise floor. The crossed control is retained but no longer discriminates.

## ddG, uncalibrated

`ddG = dA_folded - dA_unfolded`, positive meaning the mutant is less stable.

| site | dA_folded | dA_unfolded | ddG | sd(dA_folded) |
|---|---|---|---|---|
| I6A  | -10.25 | -17.00 |  6.75 | 0.32 |
| L17A |   3.56 | -13.78 | 17.34 | 0.32 |
| Y28A |   6.68 | -10.57 | 17.25 | 0.24 |

Signs are right: all three are buried and all three destabilise. Precision is
0.2-0.7 kcal/mol, comfortably under the signal. **The magnitudes are wrong**,
by roughly 3-4x against the 2-5 kcal/mol typical of a buried hydrophobic
deletion. These are not quotable as ddG values.

The failure is not sampling. Sidechain ensembles are flat by 12k sweeps, seeds
agree to 0.24, and the numbers reproduce. It is the energy model: there is no
solvation term, so deleting a buried hydrophobic pays the full vacuum vdW
contact cost with no compensating desolvation credit in the unfolded state.
That asymmetry inflates every X->Ala deletion and inflates it most for the
largest sidechains, which is the observed ordering (Leu and Tyr at 17, Ile at
7).

The Ile outlier is worth a separate look: dA_folded is *negative*, meaning the
mutant is more stable folded, which should not happen for a buried deletion.
The likely cause is that Ile6's template rotamer starts strained under this
force field, so removing it releases spurious clash energy. This is the same
suspicion recorded in the vdW blowup work and is still unproven.

## Not established

- No absolute ddG is calibrated. Do not quote one.
- `S_conf` is a first-order occupancy entropy over independent torsions and so
  an upper bound. It is only used here inside differences.
- The frozen-backbone offset of R*T*ln24 = 1.89 kcal/mol per backbone torsion
  cancels in this double difference because WT and mutant share backbone
  torsion count. It does not cancel in a single folding free energy.
