#!/bin/bash
# Full CardioSim experiment queue: main comparison (5 seeds) and ablations (3 seeds).
cd "$(dirname "$0")"
for s in 0 1 2 3 4; do
  for m in twin cox deepsurv deephit ddh rsf mmt; do
    [ -f ../runs/sim/${m}_s${s}.npz ] || python3 run_sim.py --model $m --seed $s 2>&1 | grep -v Warn
  done
done
for s in 0 1 2; do
  for m in twin_nostruct twin_mono twin_nopen twin_noipw twin_nostale twin_lastvisit twin_nosample twin_nopre; do
    [ -f ../runs/sim/${m}_s${s}.npz ] || python3 run_sim.py --model $m --seed $s 2>&1 | grep -v Warn
  done
done
echo ALLDONE
