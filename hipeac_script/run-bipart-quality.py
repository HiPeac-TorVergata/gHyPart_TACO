import input_header as input
import time
import pandas as pd
import os
import re
import datetime

current_date = datetime.date.today()

nparts = [2, 3, 4]

edgecut_output = f'../results/hipeac_bipart_quality_{input.machine_name}_t{input.num_thread}_{current_date}.csv'

ref_results = "../results/quality_comp_all_240112.csv"

data = pd.read_csv(ref_results)
original_bp2 = data['BiPart']
original_gp2 = data['gHyPart']
hedges = data['HEs']
dataset = data['dataset']
bp_policy = data['policy2']
gp_policy = data['policy1']

bipart_policy = ['RAND', 'PP', 'PLD', 'WD', 'MWD', 'DEG', 'MDEG']
our_policy_list = ['-RAND', '-HDH', '-LDH', '-LSNWH', '-HSNWH', '-LRH', '-HRH']


def prof_bipart_quality_results():
    
    cmd = f"cd ../cpu_works/BiPart && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make bipart-cpu -j"
    os.system(cmd)
    cmd = f"mkdir -p out"
    os.system(cmd)
    cmd = f"cd out && mkdir -p bipart"
    os.system(cmd)
    cmd = f"cd out/bipart && mkdir -p {current_date}"
    os.system(cmd)
    
    with open(edgecut_output, 'w') as out:
        out.write("id,dataset,k=1,k=2,k=3,hedges\n")

    with open(edgecut_output, 'a') as out:
        count = 0
        for key, value in input.input_map.items():
            file_path = input.os.path.join(input.dir_path, key)
            if count >= 0:
                out.write(str(count)+","+value+",")
                print(count, value)
                hedges = 0
                for n in nparts:
                    LOG1 = "run-bipart.log"
                    cut_result = []
                    time = []

                    cmd = f"../cpu_works/BiPart/build/lonestar/analytics/cpu/bipart/bipart-cpu -hMetisGraph "
                    cmd += f"{file_path} -t={input.num_thread} "
                    cmd += f"--numPartitions={n} "
                    cmd += f"--output=1 > {LOG1}"
                    print(cmd)

                    input.subprocess.call(cmd, shell=True)
                    
                    with open(f'{LOG1}', 'r') as file:
                        for line in file:
                            match = re.search(r'Final Edge Cut,(\d+)', line)
                            if match:
                                bipart = int(match.group(1))
                    
                            match = re.search(r'number of hedges (\d+)$', line)
                            if match:
                                hedges = int(match.group(1))
                                
                    print(f"n={n}, edge_cut={bipart}, num_hedges={hedges}")
                    out.write(f"{bipart},")
            
                out.write(f"{hedges}\n")
                out.flush()
            count += 1



if __name__ == '__main__':
    
    start_time = time.time()
    prof_bipart_quality_results()
    elapsed_time = time.time() - start_time  
    print(f"time: {elapsed_time} s, {elapsed_time / 3600} h")
