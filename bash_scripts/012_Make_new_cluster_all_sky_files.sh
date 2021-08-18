#!/bin/bash

cd ../python_scripts

nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.2 'Model_A0' > ../log_files/clu_ModelA0_frac0.2.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.1 'Model_A0' > ../log_files/clu_ModelA0_frac0.1.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.15 'Model_A0' > ../log_files/clu_ModelA0_frac0.15.log &

nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.2 'Model_A1' > ../log_files/clu_ModelA0_frac0.2.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.1 'Model_A1' > ../log_files/clu_ModelA1_frac0.1.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.15 'Model_A1' > ../log_files/clu_ModelA1_frac0.15.log &

nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.2 'Model_A2' > ../log_files/clu_ModelA2_frac0.2.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.1 'Model_A2' > ../log_files/clu_ModelA2_frac0.1.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.15 'Model_A2' > ../log_files/clu_ModelA2_frac0.15.log &

nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.2 'Model_A3' > ../log_files/clu_ModelA3_frac0.2.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.1 'Model_A3' > ../log_files/clu_ModelA3_frac0.1.log &
nohup python3 012_Make_new_cluster_file_w_changed_Lx_RF.py 0.15 'Model_A3' > ../log_files/clu_ModelA3_frac0.15.log &
