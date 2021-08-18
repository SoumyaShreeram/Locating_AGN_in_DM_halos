#!/bin/bash

cd ../python_scripts

nohup python3 001_Compute_tmm.py 0 10 > ../log_files/calTmm_z0_2_000-010.log &
nohup python3 001_Compute_tmm.py 10 100 > ../log_files/calTmm_z0_2_010-100.log &
nohup python3 001_Compute_tmm.py 100 200 > ../log_files/calTmm_z0_2_100-200.log &
nohup python3 001_Compute_tmm.py 200 300 > ../log_files/calTmm_z0_2_200-300.log &
nohup python3 001_Compute_tmm.py 300 400 > ../log_files/calTmm_z0_2_300-400.log &
nohup python3 001_Compute_tmm.py 400 500 > ../log_files/calTmm_z0_2_400-500.log &
nohup python3 001_Compute_tmm.py 500 600 > ../log_files/calTmm_z0_2_500-600.log &
nohup python3 001_Compute_tmm.py 600 700 > ../log_files/calTmm_z0_2_600-700.log &
nohup python3 001_Compute_tmm.py 700 767 > ../log_files/calTmm_z0_2_700-767.log &