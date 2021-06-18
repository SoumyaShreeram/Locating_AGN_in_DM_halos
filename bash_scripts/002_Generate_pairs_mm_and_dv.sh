#!/bin/bash

cd ../python_scripts

nohup python3 002_Generate_pairs_mm_and_dv.py 0 10 > ../log_files/mm_dv_counts_px_0-10.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 10 100 > ../log_files/mm_dv_counts_px_10-100.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 100 200 > ../log_files/mm_dv_counts_px_100-200.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 200 300 > ../log_files/mm_dv_counts_px_200-300.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 300 400 > ../log_files/mm_dv_counts_px_300-400.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 400 500 > ../log_files/mm_dv_counts_px_400-500.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 500 600 > ../log_files/mm_dv_counts_px_500-600.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 600 700 > ../log_files/mm_dv_counts_px_600-700.log &
nohup python3 002_Generate_pairs_mm_and_dv.py 700 767 > ../log_files/mm_dv_counts_px_700-767.log &