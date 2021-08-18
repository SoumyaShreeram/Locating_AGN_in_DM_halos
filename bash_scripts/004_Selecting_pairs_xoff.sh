#!/bin/bash

cd ../python_scripts

nohup python3 004_Selecting_pairs_Xoff.py 0   10  9 > ../log_files/tmm_counts_dec9_px_000-010.log &
nohup python3 004_Selecting_pairs_Xoff.py 10  100 9 > ../log_files/tmm_counts_dec9_px_010-100.log &
nohup python3 004_Selecting_pairs_Xoff.py 100 200 9 > ../log_files/tmm_counts_dec9_px_100-200.log &
nohup python3 004_Selecting_pairs_Xoff.py 200 300 9 > ../log_files/tmm_counts_dec9_px_200-300.log &
nohup python3 004_Selecting_pairs_Xoff.py 300 400 9 > ../log_files/tmm_counts_dec9_px_300-400.log &
nohup python3 004_Selecting_pairs_Xoff.py 400 500 9 > ../log_files/tmm_counts_dec9_px_400-500.log &
nohup python3 004_Selecting_pairs_Xoff.py 500 600 9 > ../log_files/tmm_counts_dec9_px_500-600.log &
nohup python3 004_Selecting_pairs_Xoff.py 600 700 9 > ../log_files/tmm_counts_dec9_px_600-700.log &
nohup python3 004_Selecting_pairs_Xoff.py 700 767 9 > ../log_files/tmm_counts_dec9_px_700-767.log &

