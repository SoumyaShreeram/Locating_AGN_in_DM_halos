#!/bin/bash

cd ../python_scripts

nohup python3 007_Regenerating_table_clu_scaling_relations.py 0 100  > ../log_files/scaling_relations_clu_files0.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 100 200  > ../log_files/scaling_relations_clu_files1.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 200 300  > ../log_files/scaling_relations_clu_files2.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 300 400  > ../log_files/scaling_relations_clu_files3.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 400 500  > ../log_files/scaling_relations_clu_files4.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 500 600  > ../log_files/scaling_relations_clu_files5.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 600 700  > ../log_files/scaling_relations_clu_files6.log &
nohup python3 007_Regenerating_table_clu_scaling_relations.py 700 767  > ../log_files/scaling_relations_clu_files7.log &