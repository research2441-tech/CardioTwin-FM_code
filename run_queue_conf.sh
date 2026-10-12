#!/bin/bash
# Sensitivity to unmeasured confounding strength gamma_u (twin vs Cox, 2 seeds).
cd "$(dirname "$0")"
for g in 0 2; do
  [ -f ../runs/cardiosim_g$g.npz ] || python3 data_sim.py --gamma_u $g --out ../runs/cardiosim_g$g.npz
  mkdir -p ../runs/sim_g$g
  for s in 0 1; do for m in twin cox mmt; do
    [ -f ../runs/sim_g$g/${m}_s$s.npz ] || CARDIO_DATA=../runs/cardiosim_g$g.npz CARDIO_RUNS=../runs/sim_g$g python3 run_sim.py --model $m --seed $s 2>&1 | grep -v Warn
  done; done
done
echo ALLDONE
