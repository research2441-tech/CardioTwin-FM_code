#!/bin/bash
# Real-cohort queue: 5-fold CV (Framingham: 1 repeat; valve cohort: 3 repeats).
cd "$(dirname "$0")"
for f in 0 1 2 3 4; do
  for m in twin cox deepsurv deephit ddh rsf mmt twin_nostruct twin_lastvisit twin_mono; do
    [ -f ../runs/real/Framingham_${m}_f${f}_r0.npz ] || python3 run_real.py --cohort Framingham --model $m --fold $f 2>&1 | grep -v -E "Warn|X\[i, k\]"
  done
done
for r in 0 1 2; do for f in 0 1 2 3 4; do
  for m in twin cox deepsurv deephit ddh rsf mmt twin_nostruct twin_lastvisit twin_mono; do
    [ -f ../runs/real/HeartValve_${m}_f${f}_r${r}.npz ] || python3 run_real.py --cohort HeartValve --model $m --fold $f --rep $r 2>&1 | grep -v -E "Warn|X\[i, k\]"
  done
done; done
echo ALLDONE
