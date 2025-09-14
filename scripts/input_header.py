import csv
# import matplotlib.pyplot as plt
import numpy as np
import random
import sys
# sys.path.append(".") # 将指定的路径添加到Python模块搜索路径列表中
# import mymodule
import re
import math
import subprocess
from subprocess import call
import argparse
# import pandas as pd
# import matplotlib
import os
# if os.environ.get('DISPLAY', '') == '':
#   matplotlib.use('Agg')
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# from matplotlib.ticker import ScalarFormatter
# from matplotlib.ticker import FuncFormatter
# from matplotlib.font_manager import FontProperties
# from matplotlib.colors import ListedColormap
# from matplotlib import colors as mcolors
# import seaborn as sns

# dir_path = '/workspace/benchmark_set/'
dir_path = '../benchmark_set/'
machine_name = "unknown"
with open("/proc/cpuinfo") as f:
    for line in f:
        if "model name" in line:
            print(line.strip())
            machine_name = line.strip().split(":")[1].split("CPU @")[0].strip()
            machine_name = machine_name.replace(" ","_").replace("(R)","").replace("(","").replace(")","")
            break
gpu_name = 'unknown'
num_thread = os.cpu_count()

try:
    # Esegui il comando nvidia-smi e cattura l'output
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        encoding="utf-8"
    )
    gpus = [line.strip() for line in result.splitlines() if line.strip()]
    if gpus:

        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            if i == 0:
                gpu_name = gpu
        gpu_name = gpu_name.replace(" ","_")
    else:
       gpu_name = 'not_present'
except FileNotFoundError:
    gpu_name = 'not_present'




input_map = {
      
            'sat14_10pipe_q0_k.cnf.dual.hgr' : 'sat_10pipe_d',
            'sat14_10pipe_q0_k.cnf.hgr' : 'sat_10pipe',
            'sat14_10pipe_q0_k.cnf.primal.hgr' : 'sat_10pipe_p',
            'sat14_11pipe_k.cnf.dual.hgr' : 'sat_11pipe_d',
            'sat14_11pipe_k.cnf.hgr' : 'sat_11pipe',
            'sat14_11pipe_k.cnf.primal.hgr' : 'sat_11pipe_p',
            'sat14_11pipe_q0_k.cnf.dual.hgr' : 'sat_11pipe_kd',
            'sat14_11pipe_q0_k.cnf.hgr' : 'sat_11pipe_k',
            'sat14_11pipe_q0_k.cnf.primal.hgr' : 'sat_11pipe_kp',
            'sat14_6s10.cnf.dual.hgr' : 'sat_6s10_d', # 1.2982,1.767137, P3 is better for node merging!
            'sat14_6s10.cnf.hgr' : 'sat_6s10',
            'sat14_6s10.cnf.primal.hgr' : 'sat_6s10_p',
            'sat14_6s11-opt.cnf.dual.hgr' : 'sat_6s11_d',
            'sat14_6s11-opt.cnf.hgr' : 'sat_6s11',
            'sat14_6s11-opt.cnf.primal.hgr' : 'sat_6s11_p',
            'sat14_6s12.cnf.dual.hgr' : 'sat_6s12_d',
            'sat14_6s12.cnf.hgr' : 'sat_6s12',
            'sat14_6s12.cnf.primal.hgr' : 'sat_6s12_p',
            
            'sat14_6s130-opt.cnf.dual.hgr' : 'sat_130_d',
            'sat14_6s130-opt.cnf.hgr' : 'sat_130',
            'sat14_6s130-opt.cnf.primal.hgr' : 'sat_130_p',
            'sat14_6s131-opt.cnf.dual.hgr' : 'sat_131_d',
            'sat14_6s131-opt.cnf.hgr' : 'sat_131',
            'sat14_6s131-opt.cnf.primal.hgr' : 'sat_131_p',
            'sat14_6s133.cnf.dual.hgr' : 'sat_133_d',
            'sat14_6s133.cnf.hgr' : 'sat_133',
            'sat14_6s133.cnf.primal.hgr' : 'sat_133_p',
            'sat14_6s153.cnf.dual.hgr' : 'sat_153_d',
            'sat14_6s153.cnf.hgr' : 'sat_153',
            'sat14_6s153.cnf.primal.hgr' : 'sat_153_p',
            'sat14_6s16.cnf.dual.hgr' : 'sat_16_d',
            'sat14_6s16.cnf.hgr' : 'sat_16',
            'sat14_6s16.cnf.primal.hgr' : 'sat_16_p',
            'sat14_6s184.cnf.dual.hgr' : 'sat_184_d',
            'sat14_6s184.cnf.hgr' : 'sat_184',
            'sat14_6s184.cnf.primal.hgr' : 'sat_184_p',
            'sat14_6s9.cnf.dual.hgr' : 'sat_9_d',
            'sat14_6s9.cnf.hgr' : 'sat_9', # 0.6139,1.0232, pattern miss prediction for k12!
            'sat14_6s9.cnf.primal.hgr' : 'sat_9_p',
            
            'sat14_9dlx_vliw_at_b_iq3.cnf.dual.hgr' : 'sat_9dlx_d',
            'sat14_9dlx_vliw_at_b_iq3.cnf.hgr' : 'sat_9dlx',
            'sat14_9dlx_vliw_at_b_iq3.cnf.primal.hgr' : 'sat_9dlx_p',
            #'sat14_9vliw_m_9stages_iq3_C1_bug7.cnf.dual.hgr' : 'sat_bug7_d', # pattern miss prediction for k12 from second iteration!
            'sat14_9vliw_m_9stages_iq3_C1_bug7.cnf.hgr' : 'sat_bug7',
            #'sat14_9vliw_m_9stages_iq3_C1_bug7.cnf.primal.hgr' : 'sat_bug7_p',
            'sat14_9vliw_m_9stages_iq3_C1_bug8.cnf.dual.hgr' : 'sat_bug8_d',
            'sat14_9vliw_m_9stages_iq3_C1_bug8.cnf.hgr' : 'sat_bug8',
            'sat14_9vliw_m_9stages_iq3_C1_bug8.cnf.primal.hgr' : 'sat_bug8_p',
            'sat14_aaai10-planning-ipc5-pathways-17-step21.cnf.dual.hgr' : 'sat_aaai_d',
            'sat14_aaai10-planning-ipc5-pathways-17-step21.cnf.hgr' : 'sat_aaai',
            'sat14_aaai10-planning-ipc5-pathways-17-step21.cnf.primal.hgr' : 'sat_aaai_p',
            
            'sat14_ACG-20-10p1.cnf.dual.hgr' : 'sat_ACG_10p_d',
            'sat14_ACG-20-10p1.cnf.hgr' : 'sat_ACG_10p',
            'sat14_ACG-20-10p1.cnf.primal.hgr' : 'sat_ACG_10p_p',
            'sat14_ACG-20-5p0.cnf.dual.hgr' : 'sat_ACG_5p0_d',
            'sat14_ACG-20-5p0.cnf.hgr' : 'sat_ACG_5p0',
            'sat14_ACG-20-5p0.cnf.primal.hgr' : 'sat_ACG_5p0_p',
            'sat14_ACG-20-5p1.cnf.dual.hgr' : 'sat_ACG_5p1_d',
            'sat14_ACG-20-5p1.cnf.hgr' : 'sat_ACG_5p1',
            'sat14_ACG-20-5p1.cnf.primal.hgr' : 'sat_ACG_5p1_p',
            'sat14_AProVE07-01.cnf.dual.hgr' : 'sat_01_d',
            'sat14_AProVE07-01.cnf.hgr' : 'sat_01',
            'sat14_AProVE07-01.cnf.primal.hgr' : 'sat_01_p',
            'sat14_AProVE07-27.cnf.dual.hgr' : 'sat_27_d',
            'sat14_AProVE07-27.cnf.hgr' : 'sat_27',
            'sat14_AProVE07-27.cnf.primal.hgr' : 'sat_27_p',
            
            'sat14_atco_enc1_opt1_05_21.cnf.dual.hgr' : 'sat_enc1_opt1_05_d',
            'sat14_atco_enc1_opt1_05_21.cnf.hgr' : 'sat_enc1_opt1_05',
            'sat14_atco_enc1_opt1_05_21.cnf.primal.hgr' : 'sat_enc1_opt1_05_p',
            'sat14_atco_enc1_opt1_10_21.cnf.dual.hgr' : 'sat_enc1_opt1_10_d',
            'sat14_atco_enc1_opt1_10_21.cnf.hgr' : 'sat_enc1_opt1_10', # 1.0092,2.1437, twc is better!
            'sat14_atco_enc1_opt1_10_21.cnf.primal.hgr' : 'sat_enc1_opt1_10_p',
            'sat14_atco_enc1_opt1_15_240.cnf.dual.hgr' : 'sat_enc1_opt1_15_d',
            'sat14_atco_enc1_opt1_15_240.cnf.hgr' : 'sat_enc1_opt1_15',
            'sat14_atco_enc1_opt1_15_240.cnf.primal.hgr' : 'sat_enc1_opt1_15_p',
            'sat14_atco_enc1_opt2_05_4.cnf.dual.hgr' : 'sat_enc1_opt2_4_d',
            'sat14_atco_enc1_opt2_05_4.cnf.hgr' : 'sat_enc1_opt2_4',
            'sat14_atco_enc1_opt2_05_4.cnf.primal.hgr' : 'sat_enc1_opt2_4_p',
            'sat14_atco_enc1_opt2_10_12.cnf.dual.hgr' : 'sat_enc1_opt2_12_d',
            'sat14_atco_enc1_opt2_10_12.cnf.hgr' : 'sat_enc1_opt2_12',
            'sat14_atco_enc1_opt2_10_12.cnf.primal.hgr' : 'sat_enc1_opt2_12_p',
            'sat14_atco_enc1_opt2_10_16.cnf.dual.hgr' : 'sat_enc1_opt2_16_d',
            'sat14_atco_enc1_opt2_10_16.cnf.hgr' : 'sat_enc1_opt2_16',
            'sat14_atco_enc1_opt2_10_16.cnf.primal.hgr' : 'sat_enc1_opt2_16_p',
            
            'sat14_atco_enc2_opt1_05_21.cnf.dual.hgr' : 'sat_enc2_opt1_05_d',
            'sat14_atco_enc2_opt1_05_21.cnf.hgr' : 'sat_enc2_opt1_05',
            'sat14_atco_enc2_opt1_05_21.cnf.primal.hgr' : 'sat_enc2_opt1_05_p',
            'sat14_atco_enc2_opt1_15_100.cnf.dual.hgr' : 'sat_enc2_opt1_15_d',
            'sat14_atco_enc2_opt1_15_100.cnf.hgr' : 'sat_enc2_opt1_15',
            'sat14_atco_enc2_opt1_15_100.cnf.primal.hgr' : 'sat_enc2_opt1_15_p',
            'sat14_atco_enc3_opt1_04_50.cnf.dual.hgr' : 'sat_enc3_opt1_50_d',
            'sat14_atco_enc3_opt1_04_50.cnf.hgr' : 'sat_enc3_opt1_50',
            'sat14_atco_enc3_opt1_04_50.cnf.primal.hgr' : 'sat_enc3_opt1_50_p',
            'sat14_atco_enc3_opt2_05_21.cnf.dual.hgr' : 'sat_enc3_opt2_21_d',
            'sat14_atco_enc3_opt2_05_21.cnf.hgr' : 'sat_enc3_opt2_21',
            'sat14_atco_enc3_opt2_05_21.cnf.primal.hgr' : 'sat_enc3_opt2_21_p',
            'sat14_atco_enc3_opt2_10_12.cnf.dual.hgr' : 'sat_enc3_opt2_12_d',
            'sat14_atco_enc3_opt2_10_12.cnf.hgr' : 'sat_enc3_opt2_12',
            'sat14_atco_enc3_opt2_10_12.cnf.primal.hgr' : 'sat_enc3_opt2_12_p',
            'sat14_atco_enc3_opt2_10_14.cnf.dual.hgr' : 'sat_enc3_opt2_14_d',
            'sat14_atco_enc3_opt2_10_14.cnf.hgr' : 'sat_enc3_opt2_14',
            'sat14_atco_enc3_opt2_10_14.cnf.primal.hgr' : 'sat_enc3_opt2_14_p',
            
            'sat14_blocks-blocks-37-1.130-NOTKNOWN.cnf.dual.hgr' : 'sat_blocks_d',
            'sat14_blocks-blocks-37-1.130-NOTKNOWN.cnf.hgr' : 'sat_blocks',
            'sat14_blocks-blocks-37-1.130-NOTKNOWN.cnf.primal.hgr' : 'sat_blocks_p',
            'sat14_bob12m09-opt.cnf.dual.hgr' : 'sat_bob-opt_d',
            'sat14_bob12m09-opt.cnf.hgr' : 'sat_bob-opt',
            'sat14_bob12m09-opt.cnf.primal.hgr' : 'sat_bob-opt_p',
            'sat14_bob12s02.cnf.dual.hgr' : 'sat_bob_d',
            'sat14_bob12s02.cnf.hgr' : 'sat_bob',
            'sat14_bob12s02.cnf.primal.hgr' : 'sat_bob_p',
            'sat14_c10bi_i.cnf.dual.hgr' : 'sat_c10_d', # 1.5919,2.1743, pattern P3 is better for node merging!
            'sat14_c10bi_i.cnf.hgr' : 'sat_c10', # 1.0616,1.7298, twc is better!
            'sat14_c10bi_i.cnf.primal.hgr' : 'sat_c10_p',
            'sat14_countbitssrl032.cnf.dual.hgr' : 'sat_count_d',
            'sat14_countbitssrl032.cnf.hgr' : 'sat_count',
            'sat14_countbitssrl032.cnf.primal.hgr' : 'sat_count_p',
            
            'sat14_ctl_3791_556_unsat_pre.cnf.dual.hgr' : 'sat_ctl_d',
            'sat14_ctl_3791_556_unsat_pre.cnf.hgr' : 'sat_ctl',
            'sat14_ctl_3791_556_unsat_pre.cnf.primal.hgr' : 'sat_ctl_p',
            'sat14_ctl_4291_567_5_unsat_pre.cnf.dual.hgr' : 'sat_ctl_5_d',
            'sat14_ctl_4291_567_5_unsat_pre.cnf.hgr' : 'sat_ctl_5',
            'sat14_ctl_4291_567_5_unsat_pre.cnf.primal.hgr' : 'sat_ctl_5_p',
            'sat14_dated-10-11-u.cnf.dual.hgr' : 'sat_dated_11_d',
            'sat14_dated-10-11-u.cnf.hgr' : 'sat_dated_11',
            'sat14_dated-10-11-u.cnf.primal.hgr' : 'sat_dated_11_p',
            'sat14_dated-10-17-u.cnf.dual.hgr' : 'sat_dated_17_d',
            'sat14_dated-10-17-u.cnf.hgr' : 'sat_dated_17',
            'sat14_dated-10-17-u.cnf.primal.hgr' : 'sat_dated_17_p',
            'sat14_E02F20.cnf.dual.hgr' : 'sat_E02F20_d',
            'sat14_E02F20.cnf.hgr' : 'sat_E02F20',
            'sat14_E02F20.cnf.primal.hgr' : 'sat_E02F20_p',
            'sat14_E02F22.cnf.dual.hgr' : 'sat_E02F22_d',
            'sat14_E02F22.cnf.hgr' : 'sat_E02F22',
            'sat14_E02F22.cnf.primal.hgr' : 'sat_E02F22_p',
            
            'sat14_gss-18-s100.cnf.dual.hgr' : 'sat_gss-18_d',
            'sat14_gss-18-s100.cnf.hgr' : 'sat_gss-18',
            'sat14_gss-18-s100.cnf.primal.hgr' : 'sat_gss-18_p',
            'sat14_gss-19-s100.cnf.dual.hgr' : 'sat_gss-19_d',
            'sat14_gss-19-s100.cnf.hgr' : 'sat_gss-19',
            'sat14_gss-19-s100.cnf.primal.hgr' : 'sat_gss-19_p',
            'sat14_gss-20-s100.cnf.dual.hgr' : 'sat_gss-20_d',
            'sat14_gss-20-s100.cnf.hgr' : 'sat_gss-20',
            'sat14_gss-20-s100.cnf.primal.hgr' : 'sat_gss-20_p',
            'sat14_gss-22-s100.cnf.dual.hgr' : 'sat_gss-22_d',
            'sat14_gss-22-s100.cnf.hgr' : 'sat_gss-22',
            'sat14_gss-22-s100.cnf.primal.hgr' : 'sat_gss-22_p',
            'sat14_hwmcc10-timeframe-expansion-k45-pdtvisns3p02-tseitin.cnf.dual.hgr' : 'sat_k45_d',
            'sat14_hwmcc10-timeframe-expansion-k45-pdtvisns3p02-tseitin.cnf.hgr' : 'sat_k45',
            'sat14_hwmcc10-timeframe-expansion-k45-pdtvisns3p02-tseitin.cnf.primal.hgr' : 'sat_k45_p',
            
            'sat14_itox_vc1130.cnf.dual.hgr' : 'sat_itox_d', # 0.4868,1.8053, pattern p1 is best for k12!
            'sat14_itox_vc1130.cnf.hgr' : 'sat_itox',
            'sat14_itox_vc1130.cnf.primal.hgr' : 'sat_itox_p',
            'sat14_k2fix_gr_rcs_w9.shuffled.cnf.dual.hgr' : 'sat_k2fix_d',
            'sat14_k2fix_gr_rcs_w9.shuffled.cnf.hgr' : 'sat_k2fix',
            'sat14_k2fix_gr_rcs_w9.shuffled.cnf.primal.hgr' : 'sat_k2fix_p',
            'sat14_manol-pipe-c10nid_i.cnf.dual.hgr' : 'sat_manol_c10_i_d',
            'sat14_manol-pipe-c10nid_i.cnf.hgr' : 'sat_manol_c10_i',
            'sat14_manol-pipe-c10nid_i.cnf.primal.hgr' : 'sat_manol_c10_i_p', # pattern miss prediction for node merging!
            'sat14_manol-pipe-c10nidw.cnf.dual.hgr' : 'sat_manol_c10_d',
            'sat14_manol-pipe-c10nidw.cnf.hgr' : 'sat_manol_c10',
            'sat14_manol-pipe-c10nidw.cnf.primal.hgr' : 'sat_manol_c10_p',
            'sat14_manol-pipe-c8nidw.cnf.dual.hgr' : 'sat_manol_c8_d',
            'sat14_manol-pipe-c8nidw.cnf.hgr' : 'sat_manol_c8',
            'sat14_manol-pipe-c8nidw.cnf.primal.hgr' : 'sat_manol_c8_p', #
            'sat14_manol-pipe-g10bid_i.cnf.dual.hgr' : 'sat_manol_g10_i_d',
            'sat14_manol-pipe-g10bid_i.cnf.hgr' : 'sat_manol_g10_i',
            'sat14_manol-pipe-g10bid_i.cnf.primal.hgr' : 'sat_manol_g10_i_p',
            
            'sat14_MD5-28-1.cnf.dual.hgr' : 'sat_MD5-28-1_d',
            'sat14_MD5-28-1.cnf.hgr' : 'sat_MD5-28-1',
            'sat14_MD5-28-1.cnf.primal.hgr' : 'sat_MD5-28-1_p',
            'sat14_MD5-28-2.cnf.dual.hgr' : 'sat_MD5-28-2_d',
            'sat14_MD5-28-2.cnf.hgr' : 'sat_MD5-28-2',
            'sat14_MD5-28-2.cnf.primal.hgr' : 'sat_MD5-28-2_p',
            'sat14_MD5-28-4.cnf.dual.hgr' : 'sat_MD5-28-4_d',
            'sat14_MD5-28-4.cnf.hgr' : 'sat_MD5-28-4',
            'sat14_MD5-28-4.cnf.primal.hgr' : 'sat_MD5-28-4_p',
            'sat14_MD5-29-4.cnf.dual.hgr' : 'sat_MD5-29-4_d',
            'sat14_MD5-29-4.cnf.hgr' : 'sat_MD5-29-4',
            'sat14_MD5-29-4.cnf.primal.hgr' : 'sat_MD5-29-4_p',
            'sat14_MD5-30-4.cnf.dual.hgr' : 'sat_MD5-30-4_d',
            'sat14_MD5-30-4.cnf.hgr' : 'sat_MD5-30-4',
            'sat14_MD5-30-4.cnf.primal.hgr' : 'sat_MD5-30-4_p',
            'sat14_MD5-30-5.cnf.dual.hgr' : 'sat_MD5-30-5_d',
            'sat14_MD5-30-5.cnf.hgr' : 'sat_MD5-30-5', # pattern p1 is best for k12!
            'sat14_MD5-30-5.cnf.primal.hgr' : 'sat_MD5-30-5_p',
            'sat14_minandmaxor128.cnf.dual.hgr' : 'sat_min_d',
            'sat14_minandmaxor128.cnf.hgr' : 'sat_min',
            'sat14_minandmaxor128.cnf.primal.hgr' : 'sat_min_p',
            
            'sat14_openstacks-p30_3.085-SAT.cnf.dual.hgr' : 'sat_p30_d',
            'sat14_openstacks-p30_3.085-SAT.cnf.hgr' : 'sat_p30',
            'sat14_openstacks-p30_3.085-SAT.cnf.primal.hgr' : 'sat_p30_p',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.025-NOTKNOWN.cnf.dual.hgr' : 'sat_p30_NOT_d',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.025-NOTKNOWN.cnf.hgr' : 'sat_p30_NOT',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.025-NOTKNOWN.cnf.primal.hgr' : 'sat_p30_NOT_p',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.085-SAT.cnf.dual.hgr' : 'sat_non_p30_d',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.085-SAT.cnf.hgr' : 'sat_non_p30',
            'sat14_openstacks-sequencedstrips-nonadl-nonnegated-os-sequencedstrips-p30_3.085-SAT.cnf.primal.hgr' : 'sat_non_p30_p',
            'sat14_post-cbmc-aes-d-r2-noholes.cnf.dual.hgr' : 'sat_post_d',
            'sat14_post-cbmc-aes-d-r2-noholes.cnf.hgr' : 'sat_post',
            'sat14_post-cbmc-aes-d-r2-noholes.cnf.primal.hgr' : 'sat_post_p',
            'sat14_post-cbmc-aes-ee-r2-noholes.cnf.dual.hgr' : 'sat_post-ee_d',
            'sat14_post-cbmc-aes-ee-r2-noholes.cnf.hgr' : 'sat_post-ee',
            'sat14_post-cbmc-aes-ee-r2-noholes.cnf.primal.hgr' : 'sat_post-ee_p',
            
            'sat14_q_query_3_L100_coli.sat.cnf.dual.hgr' : 'sat_L100_d',
            'sat14_q_query_3_L100_coli.sat.cnf.hgr' : 'sat_L100',
            'sat14_q_query_3_L100_coli.sat.cnf.primal.hgr' : 'sat_L100_p',
            'sat14_q_query_3_L150_coli.sat.cnf.dual.hgr' : 'sat_L150_d',
            'sat14_q_query_3_L150_coli.sat.cnf.hgr' : 'sat_L150',
            'sat14_q_query_3_L150_coli.sat.cnf.primal.hgr' : 'sat_L150_p',
            'sat14_q_query_3_L200_coli.sat.cnf.dual.hgr' : 'sat_L200_d',
            'sat14_q_query_3_L200_coli.sat.cnf.hgr' : 'sat_L200',
            'sat14_q_query_3_L200_coli.sat.cnf.primal.hgr' : 'sat_L200_p',
            'sat14_q_query_3_L80_coli.sat.cnf.dual.hgr' : 'sat_L80_d',
            'sat14_q_query_3_L80_coli.sat.cnf.hgr' : 'sat_L80',
            'sat14_q_query_3_L80_coli.sat.cnf.primal.hgr' : 'sat_L80_p',
            
            'sat14_SAT_dat.k100-24_1_rule_1.cnf.dual.hgr' : 'sat_k100_1d',
            'sat14_SAT_dat.k100-24_1_rule_1.cnf.hgr' : 'sat_k100_1',
            'sat14_SAT_dat.k100-24_1_rule_1.cnf.primal.hgr' : 'sat_k100_1p',
            'sat14_SAT_dat.k100-24_1_rule_2.cnf.dual.hgr' : 'sat_k100_2d',
            'sat14_SAT_dat.k100-24_1_rule_2.cnf.hgr' : 'sat_k100_2',
            'sat14_SAT_dat.k100-24_1_rule_2.cnf.primal.hgr' : 'sat_k100_2p',
            'sat14_SAT_dat.k70-24_1_rule_1.cnf.dual.hgr' : 'sat_k70_d',
            'sat14_SAT_dat.k70-24_1_rule_1.cnf.hgr' : 'sat_k70', # 0.1261,1.0135, pattern P1 is better for k12! completely wrong!
            'sat14_SAT_dat.k70-24_1_rule_1.cnf.primal.hgr' : 'sat_k70_p',
            'sat14_SAT_dat.k75-24_1_rule_3.cnf.dual.hgr' : 'sat_k75_d',
            'sat14_SAT_dat.k75-24_1_rule_3.cnf.hgr' : 'sat_k75',
            'sat14_SAT_dat.k75-24_1_rule_3.cnf.primal.hgr' : 'sat_k75_p',
            'sat14_SAT_dat.k80-24_1_rule_1.cnf.dual.hgr' : 'sat_k80_d',
            'sat14_SAT_dat.k80-24_1_rule_1.cnf.hgr' : 'sat_k80',
            'sat14_SAT_dat.k80-24_1_rule_1.cnf.primal.hgr' : 'sat_k80_p',
            'sat14_SAT_dat.k85-24_1_rule_2.cnf.dual.hgr' : 'sat_k85_2d',
            'sat14_SAT_dat.k85-24_1_rule_2.cnf.hgr' : 'sat_k85_2',
            'sat14_SAT_dat.k85-24_1_rule_2.cnf.primal.hgr' : 'sat_k85_2p',
            'sat14_SAT_dat.k85-24_1_rule_3.cnf.dual.hgr' : 'sat_k85_3d',
            'sat14_SAT_dat.k85-24_1_rule_3.cnf.hgr' : 'sat_k85_3',
            'sat14_SAT_dat.k85-24_1_rule_3.cnf.primal.hgr' : 'sat_k85_3p',
                        
            'sat14_SAT_dat.k90.debugged.cnf.dual.hgr' : 'sat_k90_d',
            'sat14_SAT_dat.k90.debugged.cnf.hgr' : 'sat_k90',
            'sat14_SAT_dat.k90.debugged.cnf.primal.hgr' : 'sat_k90_p',
            'sat14_SAT_dat.k95-24_1_rule_3.cnf.dual.hgr' : 'sat_k95_d',
            'sat14_SAT_dat.k95-24_1_rule_3.cnf.hgr' : 'sat_k95',
            'sat14_SAT_dat.k95-24_1_rule_3.cnf.primal.hgr' : 'sat_k95_p',
            'sat14_slp-synthesis-aes-top29.cnf.dual.hgr' : 'sat_slp_d',
            'sat14_slp-synthesis-aes-top29.cnf.hgr' : 'sat_slp',
            'sat14_slp-synthesis-aes-top29.cnf.primal.hgr' : 'sat_slp_p', # 1.2689,2.5864, pattern miss prediction for k12!
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.030-NOTKNOWN.cnf.dual.hgr' : 'sat_030_d',
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.030-NOTKNOWN.cnf.hgr' : 'sat_030',
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.030-NOTKNOWN.cnf.primal.hgr' : 'sat_030_p',
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.050-NOTKNOWN.cnf.dual.hgr' : 'sat_050_d',
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.050-NOTKNOWN.cnf.hgr' : 'sat_050',
            'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.050-NOTKNOWN.cnf.primal.hgr' : 'sat_050_p',
            
            'sat14_UCG-15-10p0.cnf.dual.hgr' : 'sat_UCG-10p0_d',
            'sat14_UCG-15-10p0.cnf.hgr' : 'sat_UCG-10p0',
            'sat14_UCG-15-10p0.cnf.primal.hgr' : 'sat_UCG-10p0_p',
            'sat14_UCG-15-10p1.cnf.dual.hgr' : 'sat_UCG-10p1_d',
            'sat14_UCG-15-10p1.cnf.hgr' : 'sat_UCG-10p1',
            'sat14_UCG-15-10p1.cnf.primal.hgr' : 'sat_UCG-10p1_p',
            'sat14_UCG-20-5p0.cnf.dual.hgr' : 'sat_UCG-5p0_d',
            'sat14_UCG-20-5p0.cnf.hgr' : 'sat_UCG-5p0',
            'sat14_UCG-20-5p0.cnf.primal.hgr' : 'sat_UCG-5p0_p',
            'sat14_UR-15-10p1.cnf.dual.hgr' : 'sat_UR-15_d',
            'sat14_UR-15-10p1.cnf.hgr' : 'sat_UR-15',
            'sat14_UR-15-10p1.cnf.primal.hgr' : "sat_UR-15_p",
            'sat14_UR-20-5p0.cnf.dual.hgr' : 'sat_UR-20_d',
            'sat14_UR-20-5p0.cnf.hgr' : 'sat_UR-20', # 0.7382,1.0000, baseline is best
            'sat14_UR-20-5p0.cnf.primal.hgr' : 'sat_UR-20_p',
            'sat14_UTI-20-10p1.cnf.dual.hgr' : 'sat_UTI_d',
            'sat14_UTI-20-10p1.cnf.hgr' : 'sat_UTI',
            'sat14_UTI-20-10p1.cnf.primal.hgr' : 'sat_UTI_p',
            'sat14_velev-vliw-uns-2.0-uq5.cnf.dual.hgr' : 'sat_2.0_d', # 44.1638,55.2703, pattern miss prediction for k12!
            'sat14_velev-vliw-uns-2.0-uq5.cnf.hgr' : 'sat_2.0',
            'sat14_velev-vliw-uns-2.0-uq5.cnf.primal.hgr' : 'sat_2.0_p',
            'sat14_velev-vliw-uns-4.0-9.cnf.dual.hgr' : 'sat_4.0-9_d',
            'sat14_velev-vliw-uns-4.0-9.cnf.hgr' : 'sat_4.0-9',
            'sat14_velev-vliw-uns-4.0-9.cnf.primal.hgr' : 'sat_4.0-9_p',

############################################################################################################################
            '192bit.mtx.hgr' : '192bit',
            '2cubes_sphere.mtx.hgr' : '2cubes',
            '2D_54019_highK.mtx.hgr' : '2D',
            '3Dspectralwave2.mtx.hgr' : '3D',
            'ABACUS_shell_hd.mtx.hgr' : 'ABACUS',
            'af23560.mtx.hgr' : 'af23560',
            'af_shell1.mtx.hgr' : 'af_shell',
            'airfoil_2d.mtx.hgr' : 'airfoil',
            'Andrews.mtx.hgr' : 'Andrews', # 0.7012,1.0782, p3 for k123_k5 is best
            'appu.mtx.hgr' : 'appu',
            'as-22july06.mtx.hgr' : 'as-22',
            # 'as-caida.mtx.hgr' : 'as-caida', # 0.1494,1.1773, pattern P1 is best for k12! completely wrong!
            'astro-ph.mtx.hgr' : 'astro',
            'atmosmodj.mtx.hgr' : 'atmos',
            'av41092.mtx.hgr' : 'av41092',
            'barrier2-1.mtx.hgr' : 'barrier',
            'Baumann.mtx.hgr' : 'Baumann',
            'bayer04.mtx.hgr' : 'bayer04',
            'bbmat.mtx.hgr' : 'bbmat',
            'bcsstk29.mtx.hgr' : 'bcsstk',
            'BenElechi1.mtx.hgr' : 'Ben',
            'bibd_49_3.mtx.hgr' : 'bibd', # 1.0141,1.5292, pattern is correct, but twc is better.
            'bips07_1998.mtx.hgr' : 'bips',
            'bloweya.mtx.hgr' : 'bloweya', # 12.0438,15.7791, pattern miss prediction for k12!
            'bundle1.mtx.hgr' : 'bundle', # 1.3734,2.4892, pattern is correct, but twc is better.
            'c-55.mtx.hgr' : 'c-55',
            'c-61.mtx.hgr' : 'c-61',
            'c-64.mtx.hgr' : 'c-64',
            'ca-CondMat.mtx.hgr' : 'CondMat',
            'cage10.mtx.hgr' : 'cage10',
            'ccc.mtx.hgr' : 'ccc',
            'cfd1.mtx.hgr' : 'cfd', # 0.7236,1.0468, p3 for k123_k5 is best
            
            'Chem97Zt.mtx.hgr' : 'Chem97Zt', # 1.4196,2.7327, pattern miss prediction for node merging!
            'circuit_3.mtx.hgr' : 'circuit',
            'ckt11752_dc_1.mtx.hgr' : 'ckt',
            'cnr-2000.mtx.hgr' : 'cnr',
            'conf5_4-8x8-05.mtx.hgr' : 'conf5_4',
            'copter1.mtx.hgr' : 'copter',
            'coupled.mtx.hgr' : 'coupled', # 1.7535,2.4847, pattern miss prediction for node merging!
            'crashbasis.mtx.hgr' : 'crash',
            'cryg10000.mtx.hgr' : 'cryg',
            'crystk02.mtx.hgr' : 'crystk',
            'd_pretok.mtx.hgr' : 'd_pretok',

            'dac2012_superblue11.hgr' : 'dac11',
            'dac2012_superblue14.hgr' : 'dac14',
            'dac2012_superblue16.hgr' : 'dac16',
            'dac2012_superblue2.hgr' : 'dac2',
            'dac2012_superblue3.hgr' : 'dac3',
            'dac2012_superblue6.hgr' : 'dac6',
            'dac2012_superblue7.hgr' : 'dac7',

            'deltaX.mtx.hgr' : 'delta',
            'denormal.mtx.hgr' : 'denormal',
            'dictionary28.mtx.hgr' : 'dict',
            'dielFilterV2clx.mtx.hgr' : 'diel',
            'Dubcova2.mtx.hgr' : 'cova2',
            'ecl32.mtx.hgr' : 'ecl32',
            'Emilia_923.mtx.hgr' : 'Emilia',
            'epb1.mtx.hgr' : 'epb1',
            'ESOC.mtx.hgr' : 'ESOC',
            'EternityII_A.mtx.hgr' : 'Eternity',
            'ex19.mtx.hgr' : 'ex19',
            'F2.mtx.hgr' : 'F2',
            'fd18.mtx.hgr' : 'fd18',
            'fem_filter.mtx.hgr' : 'fem',
            'finan512.mtx.hgr' : 'finan',
            'flower_7_4.mtx.hgr' : 'flower',
            'foldoc.mtx.hgr' : 'foldoc',
            'Franz11.mtx.hgr' : 'Franz',
            
            'G2_circuit.mtx.hgr' : 'G2',
            'g7jac040sc.mtx.hgr' : 'g7jac',
            'garon2.mtx.hgr' : 'garon',
            'gearbox.mtx.hgr' : 'gearbox',
            'gemat1.mtx.hgr' : 'gemat',
            'graphics.mtx.hgr' : 'graphics',
            'gyro.mtx.hgr' : 'gyro',
            'H2O.mtx.hgr' : 'H2O',
            'HEP-th.mtx.hgr' : 'HEP',
            'HTC_336_9129.mtx.hgr' : 'HTC', # 2.8498,3.6195, pattern is correct, but twc is better.
            'hvdc1.mtx.hgr' : 'hvdc1',
            'IG5-17.mtx.hgr' : 'IG5',
            'Ill_Stokes.mtx.hgr' : 'Ill',
            'image_interp.mtx.hgr' : 'interp',
            'IMDB.mtx.hgr' : 'IMDB',

            'ISPD98_ibm01.hgr' : 'ibm01',
            'ISPD98_ibm02.hgr' : 'ibm02',
            'ISPD98_ibm03.hgr' : 'ibm03',
            'ISPD98_ibm04.hgr' : 'ibm04',
            'ISPD98_ibm05.hgr' : 'ibm05',
            'ISPD98_ibm06.hgr' : 'ibm06',
            'ISPD98_ibm07.hgr' : 'ibm07',
            'ISPD98_ibm08.hgr' : 'ibm08',
            'ISPD98_ibm09.hgr' : 'ibm09',
            'ISPD98_ibm10.hgr' : 'ibm10', # 0.7563,1.0686, p3 for k123_k5 is best
            'ISPD98_ibm11.hgr' : 'ibm11',
            'ISPD98_ibm12.hgr' : 'ibm12',
            'ISPD98_ibm13.hgr' : 'ibm13', # 0.6304,1.0068, p3 for k123_k5 is best
            'ISPD98_ibm14.hgr' : 'ibm14',
            'ISPD98_ibm15.hgr' : 'ibm15',
            'ISPD98_ibm16.hgr' : 'ibm16',
            
            'kim1.mtx.hgr' : 'kim1',
            'kkt_power.mtx.hgr' : 'kkt',
            'laminar_duct3D.mtx.hgr' : 'duct3D',
            'lhr14.mtx.hgr' : 'lhr14',
            'li.mtx.hgr' : 'li',
            'light_in_tissue.mtx.hgr' : 'tissue',
            'Lin.mtx.hgr' : 'Lin',
            'lp_nug20.mtx.hgr' : 'lp_nug',
            'lp_pds_20.mtx.hgr' : 'lp_pds', # 0.6388,1.0202, p3 for k123_k5 is best
            'lung2.mtx.hgr' : 'lung',
            'm14b.mtx.hgr' : 'm14b',
            'mac_econ_fwd500.mtx.hgr' : 'mac',
            'Maragal_6.mtx.hgr' : 'Maragal', # 2.6439,3.4988, pattern is ok, but twc is better.
            'mc2depi.mtx.hgr' : 'mc2depi',
            'mixtank_new.mtx.hgr' : 'mixtank',
            'mono_500Hz.mtx.hgr' : 'mono',
            'mri1.mtx.hgr' : 'mri1',
            'msc10848.mtx.hgr' : 'msc', # 1.1365,1.7313, pattern is ok, but twc is better.
            'msdoor.mtx.hgr' : 'msdoor',
            'mult_dcop_01.mtx.hgr' : 'mult_dcop', # 3.7165,17.4473, pattern miss prediction for k12!
            
            'nasasrb.mtx.hgr' : 'nasasrb',
            'nd12k.mtx.hgr' : 'nd12k',
            'net100.mtx.hgr' : 'net100',
            'nopoly.mtx.hgr' : 'nopoly',
            'NotreDame_actors.mtx.hgr' : 'actors',
            'NotreDame_www.mtx.hgr' : 'www',
            'obstclae.mtx.hgr' : 'obstclae',
            'olafu.mtx.hgr' : 'olafu',
            'opt1.mtx.hgr' : 'opt1',
            'Oregon-1.mtx.hgr' : 'Oregon',
            'p2p-Gnutella25.mtx.hgr' : 'p2p',
            'para-4.mtx.hgr' : 'para', # 2.1687,3.2794, pattern miss prediction for node merging!
            'parabolic_fem.mtx.hgr' : 'parabolic',
            'Pd_rhs.mtx.hgr' : 'Pd', # 1.0603,2.1278, pattern miss prediction for twc!
            'pdb1HYS.mtx.hgr' : 'pdb',
            'pds-90.mtx.hgr' : 'pds',
            'pesa.mtx.hgr' : 'pesa',
            'PGPgiantcompo.mtx.hgr' : 'PGP',
            'pkustk11.mtx.hgr' : 'pkustk',
            'poisson3Db.mtx.hgr' : 'poisson3D',
            'poli3.mtx.hgr' : 'poli3',
            'powersim.mtx.hgr' : 'powersim',
            'pre2.mtx.hgr' : 'pre2',
            'Pres_Poisson.mtx.hgr' : 'Pres',
            'psse2.mtx.hgr' : 'psse2',
            'qa8fk.mtx.hgr' : 'qa8fk',
            'rajat26.mtx.hgr' : 'rajat',
            'Reuters911.mtx.hgr' : 'Reuters', # 1.1642,1.5886, pattern miss prediction for node merging!
            'RFdevice.mtx.hgr' : 'RFdevice',
            'rgg_n_2_18_s0.mtx.hgr' : 'rgg',
            'rim.mtx.hgr' : 'rim',
            'rma10.mtx.hgr' : 'rma',
            'Rucci1.mtx.hgr' : 'Rucci',
            's4dkt3m2.mtx.hgr' : 's4dkt',

            'scfxm1-2r.mtx.hgr' : 'scfxm',
            'scircuit.mtx.hgr' : 'scircuit',
            'shallow_water2.mtx.hgr' : 'shallow',
            'shermanACb.mtx.hgr' : 'sherman',
            'ship_001.mtx.hgr' : 'ship',
            'shock-9.mtx.hgr' : 'shock',
            'shyy161.mtx.hgr' : 'shyy',
            'skirt.mtx.hgr' : 'skirt',
            
            'sme3Db.mtx.hgr' : 'sme3D',
            'soc-sign-epinions.mtx.hgr' : 'soc-sign', # 1.0328,7.5877, pattern is correct, but twc is better.
            'sparsine.mtx.hgr' : 'sparsine',
            'spmsrtls.mtx.hgr' : 'spmsrtls',
            'std1_Jac3.mtx.hgr' : 'std1_Jac3',
            'stokes128.mtx.hgr' : 'stokes',
            'ted_A.mtx.hgr' : 'ted_A',
            'TF16.mtx.hgr' : 'TF16',
            'thermal1.mtx.hgr' : 'thermal',
            'thermomech_TC.mtx.hgr' : 'therm',
            'tmt_unsym.mtx.hgr' : 'tmt',
            'tomographic1.mtx.hgr' : 'tomogra',
            'torso3.mtx.hgr' : 'torso',
            'Trec14.mtx.hgr' : 'Trec',
            'Trefethen_20000.mtx.hgr' : 'Trefe',
            'TSOPF_FS_b162_c3.mtx.hgr' : 'TSOPF', # 8.5151,18.4893, pattern miss prediction for k12!

            'us04.mtx.hgr' : 'us04', # 10.3073,16.6290, pattern is correct, but twc is better.
            'usroads.mtx.hgr' : 'usroads',
            'venkat01.mtx.hgr' : 'venkat',
            'vibrobox.mtx.hgr' : 'vibro',
            'viscoplastic2.mtx.hgr' : 'plastic',
            'viscorocks.mtx.hgr' : 'rocks',
            'waveguide3D.mtx.hgr' : 'wave3D',
            'wathen120.mtx.hgr' : 'wathen120',
            'water_tank.mtx.hgr' : 'water',
            'wang4.mtx.hgr' : 'wang4',
            'xenon2.mtx.hgr' : 'xenon',
            'Zhao2.mtx.hgr' : 'Zhao',
            'G67.mtx.hgr' : 'G67',
            
            'af_4_k101.mtx.hgr' : 'af101', 
            'ecology1.mtx.hgr' : 'eco1', 
            'ISPD98_ibm18.hgr' : 'ibm18', 
            'ISPD98_ibm17.hgr' : 'ibm17', 
            'webbase-1M.mtx.hgr' : 'web1M', 
            'nlpkkt120.mtx.hgr' : 'nl120', 
            'sat14_series_bug7_primal.hgr' : 'sat_p', 
            'wb-edu.mtx.hgr' : 'wbedu', 
            'Stanford.mtx.hgr' : 'Stanf', 
            'human_gene2.mtx.hgr' : 'gene2', 
            # 'gupta3.mtx.hgr' : 'gupta', 
            'Hamrle3.mtx.hgr' : 'Hamr3', 
            'StocF-1465.mtx.hgr' : 'StocF', 
            'language.mtx.hgr' : 'lang', 
            'kron_g500-logn16.mtx.hgr' : 'kron', 
            'case39.mtx.hgr' : 'case39', # misprediction for k12 pattern!
            'dac2012_superblue9.hgr' : 'dac9', 
            'dac2012_superblue12.hgr' : 'dac12', 
            'dac2012_superblue19.hgr' : 'dac19', 
            'trans4.mtx.hgr' : 'tran4', 
            'Chebyshev4.mtx.hgr' : 'Cheb4', 
            'sat14_series_bug7_dual.hgr' : 'sat_d', 
            # 'RM07R.mtx.hgr' : 'RM07R',
            'HV15R.mtx.hgr' : 'HV15R',
            'gupta2.mtx.hgr' : 'gupta2', 
            'ASIC_680k.mtx.hgr' : 'ASIC',
            'ASIC_320k.mtx.hgr' : 'ASIC320k',
            'wiki-Talk.mtx.hgr' : 'wiki-Talk',
            'com-Youtube.mtx.hgr' : 'com-Youtube',
            
            'Freescale1.mtx.hgr' : 'Freescal',
            'roadNet-CA.mtx.hgr' : 'roadNet-CA',
            # 'road_usa.mtx.hgr' : 'road_usa',
            'delaunay_n23.mtx.hgr' : 'delaunay23',
            # 'com-LiveJournal.mtx.hgr' : 'com-LiveJournal',
            'coPapersDBLP.mtx.hgr' : 'coPapersDBLP',
            # 'us04.mtx.hgr' : 'us04', # 10.3073,16.6290, pattern is correct, but twc is better.
            'dc1.mtx.hgr' : 'dc1',
            'channel-500x100x100-b050.mtx.hgr' : 'channel',
            
            # 'sat14_esawn_uw3.debugged.cnf.dual.hgr' : 'sat_uw3_d',
            # 'sat14_sv-comp19_prop-reachsafety.barrier_3t_true-unreach-call.i-witness.cnf.dual.hgr' : 'sat_barrier_d',
            # 'sat14_sin.c.75.smt2-cvc4-sc2016.cnf.dual.hgr' : 'sat_sin_d',
            # 'sat14_velev-npe-1.0-9dlx-b71.cnf.dual.hgr' : 'sat_npe_d',
            # 'circuit5M.mtx.hgr' : 'circuit5M',
            # 'sat14_zfcp-2.8-u2-nh.cnf.dual.hgr' : 'sat_zfcp_d',
            # 'G3_circuit.mtx.hgr' : 'G3',
            # 'bloweybl.mtx.hgr' : 'bloweybl',
            'as-Skitter.mtx.hgr' : 'as-Skitter',
            'ASIC_100k.mtx.hgr' : 'ASIC_100k',
            'dc2.mtx.hgr' : 'dc2',
            'trans5.mtx.hgr' : 'tran5',
}

representatives = {
            'af_4_k101.mtx.hgr' : 'af101', 
            'ecology1.mtx.hgr' : 'eco1', 
            'ISPD98_ibm18.hgr' : 'ibm18', 
            'ISPD98_ibm17.hgr' : 'ibm17', 
            'webbase-1M.mtx.hgr' : 'web1M', 
            'nlpkkt120.mtx.hgr' : 'nl120', 
            'sat14_series_bug7_primal.hgr' : 'sat_p', 
            'wb-edu.mtx.hgr' : 'wbedu', 
            'Stanford.mtx.hgr' : 'Stanf', 
            'human_gene2.mtx.hgr' : 'gene2', 
            'sat14_series_bug7_dual.hgr' : 'sat_d', 
            'gupta3.mtx.hgr' : 'gupta', 
            'Chebyshev4.mtx.hgr' : 'Cheb4', 
            'Hamrle3.mtx.hgr' : 'Hamr3', 
            'StocF-1465.mtx.hgr' : 'StocF', 
            'trans4.mtx.hgr' : 'tran4', 
            'dac2012_superblue9.hgr' : 'dac9', 
            'language.mtx.hgr' : 'lang', 
            'kron_g500-logn16.mtx.hgr' : 'kron', 
            'case39.mtx.hgr' : 'case39', 
            'dac2012_superblue12.hgr' : 'dac12', 
            'dac2012_superblue19.hgr' : 'dac19', 
            'RM07R.mtx.hgr' : 'RM07R', 
            'us04.mtx.hgr' : 'us04', 
}

sampling_datasets = {
            'sat14_atco_enc3_opt2_05_21.cnf.primal.hgr' : 'sat_enc3_opt2_21_p',
            'sat14_SAT_dat.k100-24_1_rule_1.cnf.hgr' : 'sat_k100_1',
            'sat14_blocks-blocks-37-1.130-NOTKNOWN.cnf.hgr' : 'sat_blocks',
            'sat14_q_query_3_L200_coli.sat.cnf.primal.hgr' : 'sat_L200_p',
            'sat14_9vliw_m_9stages_iq3_C1_bug8.cnf.primal.hgr' : 'sat_bug8_p',
            'language.mtx.hgr' : 'lang', 
            'wb-edu.mtx.hgr' : 'wbedu', 
            
            # 'dac2012_superblue12.hgr' : 'dac12', 
            # 'Stanford.mtx.hgr' : 'Stanf', 
            'sat14_sin.c.75.smt2-cvc4-sc2016.cnf.dual.hgr' : 'sat_sin_d', 
            'trans4.mtx.hgr' : 'tran4', 
            'us04.mtx.hgr' : 'us04', 
            'sat14_11pipe_k.cnf.dual.hgr' : 'sat_11pipe_d', 
            'Chebyshev4.mtx.hgr' : 'Cheb4', 
            
            # 'sat14_9vliw_m_9stages_iq3_C1_bug8.cnf.hgr' : 'sat_bug8',
            # 'sat14_11pipe_q0_k.cnf.hgr' : 'sat_11pipe_k',
            # 'sat14_10pipe_q0_k.cnf.hgr' : 'sat_10pipe',
            # 'sat14_velev-vliw-uns-2.0-uq5.cnf.hgr' : 'sat_2.0',
            # 'sat14_velev-vliw-uns-2.0-uq5.cnf.primal.hgr' : 'sat_2.0_p',
            # 'sat14_11pipe_q0_k.cnf.primal.hgr' : 'sat_11pipe_kp',
            # 'sat14_10pipe_q0_k.cnf.primal.hgr' : 'sat_10pipe_p',
            # 'atmosmodj.mtx.hgr' : 'atmos',
            # 'sat14_transport-transport-city-sequential-25nodes-1000size-3degree-100mindistance-3trucks-10packages-2008seed.030-NOTKNOWN.cnf.dual.hgr' : 'sat_030_d',
}

# 'sls.mtx.hgr' : 'sls', # core dump!
policy_map = {
             'G67.mtx.hgr' : '--PLD',
             'af_4_k101.mtx.hgr' : '--WD', 
             'ecology1.mtx.hgr' : '--PLD', 
             'ISPD98_ibm18.hgr' : '--PP', 
             'ISPD98_ibm17.hgr' : '--RAND', 
             
             'webbase-1M.mtx.hgr' : '--RAND', 
             'nlpkkt120.mtx.hgr' : '--PLD', 
             'sat14_series_bug7_primal.hgr' : '--MWD', 
             'wb-edu.mtx.hgr' : '--RAND', 
             'Stanford.mtx.hgr' : '--DEG', 
             'human_gene2.mtx.hgr' : '--PLD', 
             'sat14_series_bug7_dual.hgr' : '--PP', 
             
             'gupta3.mtx.hgr' : '--DEG', 
             'Chebyshev4.mtx.hgr' : '--PP', 
             'Hamrle3.mtx.hgr' : '--PLD', 
             'StocF-1465.mtx.hgr' : '--RAND', 
             'trans4.mtx.hgr' : '--MWD', 
             
             'dac2012_superblue9.hgr' : '--MDEG', 
             'language.mtx.hgr' : '--MWD', 
             'kron_g500-logn16.mtx.hgr' : '--DEG', 
             'case39.mtx.hgr' : '--PP', 
             'dac2012_superblue12.hgr' : '--PP', 
             'dac2012_superblue19.hgr' : '--PLD', 
             
             'RM07R.mtx.hgr' : '--MDEG',
             'us04.mtx.hgr' : '--PP',
             }

input_list = os.listdir(dir_path)

geom = []

mean = []

benchmark_num = len(input_map)


ncu_metrics = {
    'l2_tex_read_throughput' : 'lts__t_sectors_srcunit_tex_op_read.sum.per_second', # seems work
    'l2_tex_read_transactions' : 'lts__t_sectors_srcunit_tex_op_read.sum', # work
    'l2_tex_request_read' : 'lts__t_requests_srcunit_tex_op_read.sum',
    'l2_tex_hit_rate' : 'lts__t_sector_hit_rate.pct',
    'l1_tex_hit_rate' : 'l1tex__t_sector_hit_rate.pct',
    'l2_tex_read_hit_rate' : 'lts__t_sector_op_read_hit_rate.pct',
    'gld_transactions_per_request': 'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio',
    'gld_throughput': "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    'global_load_requests' : "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    'gld_transactions' : "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    'l1_tex_requests' : 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'gld_efficiency' : 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
    'group_l1_metric_table' : 'group:memory__first_level_cache_table',
    'group_l2_metric_table' : 'group:memory__l2_cache_table',

    'l2_achieved_glb_sectors_request' : 'memory_l2_theoretical_sectors_global',
    'l2_ideal_glb_sectors_request' : 'memory_l2_theoretical_sectors_global_ideal',

    'warp_issue_stalled' : 'smsp__pcsamp_warps_issue_stalled_long_scoreboard',
    'warp_stalled_not_issue' : 'smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued',
    # Ratio of the average active threads per warp to the maximum number of threads per warp supported on a SM
    'warp_execution_efficiency' : 'smsp__thread_inst_executed_per_inst_executed.ratio',
    '# of warp-level insts' : 'inst_executed', # sm__inst_executed.sum
    '# of thread-level insts' : 'thread_inst_executed', # smsp__thread_inst_executed.sum
    'sm_efficiency' : 'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed',
    'avg_sm_insts' : 'sm__inst_executed.avg',
    'sum_sm_insts' : 'sm__inst_executed.sum',
    'dram_utilization' : 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'memory_bandwidth' : 'dram__bytes.sum.per_second',
    'kernel_time' : 'gpu__time_duration.sum',
    'achieved_occupancy' : 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'achieved_act_warps_perSM' : 'sm__warps_active.avg.per_cycle_active'
}

# dir_path = '/home/wuzhenlin/workspace/synthetic_hypergraphs/'

synthetics = {
    'hypergraph0.hgr' : 'myhgr0',
}
