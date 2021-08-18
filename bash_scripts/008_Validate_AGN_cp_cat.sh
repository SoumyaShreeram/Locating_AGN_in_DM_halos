#!/bin/bash

cd ../python_scripts

nohup python3 008_Find_single_AGN_pairs.py 0 50    0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p000-050_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 50 100  0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p050-100_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 100 150 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p100-150_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 150 200 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p150-200_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 200 250 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p200-250_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 250 300 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p250-300_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 300 350 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p300-350_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 350 400 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p350-400_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 400 450 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p400-450_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 450 500 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p450-500_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 500 550 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p500-550_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 550 600 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p550-600_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 600 650 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p600-650_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 650 700 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p650-700_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 700 750 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p700-750_noModel.log &
nohup python3 008_Find_single_AGN_pairs.py 750 767 0.15 1.2 1.8 0.1 0.13  > ../log_files/agn_p750-767_noModel.log &

#nohup python3 008_Find_single_AGN_pairs.py 700 750 0.10 0.6 1.2 0.07 0.09  > ../log_files/agn_FCP0.1_p700-750_tmm0.6.log &
#nohup python3 008_Find_single_AGN_pairs.py 700 750 0.10 3.7 4.4 0.0 0.03  > ../log_files/agn_FCP0.1_p700-750_tmm3.7.log &


