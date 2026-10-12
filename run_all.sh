#!/bin/bash
# End-to-end reproduction of every experiment, table and figure.
set -e
cd "$(dirname "$0")"
mkdir -p ../runs/sim ../runs/real ../paper/figs
[ -f ../runs/cardiosim.npz ] || python3 data_sim.py --out ../runs/cardiosim.npz
bash run_queue.sh
bash run_queue_real.sh
bash run_queue_conf.sh
python3 analyze_all.py sim Framingham HeartValve sens
python3 fig_arch.py
python3 fig_traj.py 1
python3 fig_results.py
