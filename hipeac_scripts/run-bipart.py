# -------------------------------------------------------------------------
# This is the Tor Vergata team version of run-bipart.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# For the original version of the code please refer to:
# https://github.com/zlwu92/gHyPart_TACO/blob/master/scripts/run-bipart.py
# -------------------------------------------------------------------------
import input_header as input
import time
import os
import re
import datetime
import questionary
import pandas as pd

current_date = datetime.date.today()
nparts = [2, 3, 4]

print(input.cpu_name)

bipart_perf_results         = f'../hipeac_results/hipeac_bipart_{input.cpu_name}_t{input.num_thread}_{current_date}.csv'
bipart_perf_best_results    = f'../hipeac_results/hipeac_bipart_best_{input.cpu_name}_t{input.num_thread}_{current_date}.csv'

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

# -------------------------------------------------------------------------
# Build and run the code. For each run the log is read and the information
# about the performance of the partitions found in terms of execution time 
# is saved as an entry of a .csv file. All the results are saved under 
# hipeac_results/ directory. 
# -------------------------------------------------------------------------
def prof_bipart_results(mode):
    cmd = f"cd ../cpu_works/BiPart && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make bipart-cpu -j"
    os.system(cmd)
    cmd = f"mkdir -p out"
    os.system(cmd)
    cmd = f"cd out && mkdir -p bipart"
    os.system(cmd)
    cmd = f"cd out/bipart && mkdir -p {current_date}"
    os.system(cmd)


    policy = ""

    # Select the correct output file
    match (mode):
        case ("best policy"):
            outfile = bipart_perf_results
        case ("standard"):
            outfile = bipart_perf_best_results
        case _:
            raise ValueError("Invalid module/mode combination")


    print("Saving results on ", outfile)


    with open(outfile, 'a') as out:
        count = 0
        for key, value in input.input_map.items():
            file_path = input.os.path.join(input.dataset_relative_path, key)
            
            if(mode == "best policy"):
                policy = f"--{bp_policy[count]}"    
            
            if count >= 0:
                out.write(str(count)+","+value+",")
                print(count, value)
                for n in nparts:
                    LOG1 = "run-bipart.log"
                    output_file = f"out/bipart/{current_date}/{value}_{n}.part"
                    print("Results saved on ",output_file)
                    cmd = f"../cpu_works/BiPart/build/lonestar/analytics/cpu/bipart/bipart-cpu -hMetisGraph "
                    cmd += f"{file_path} -t={input.num_thread} "
                    cmd += f"{policy} "
                    cmd += f"--numPartitions={n} "
                    cmd += f"--output=1 --outputFile={output_file} > {LOG1}"
                    print(cmd)
                    input.subprocess.call(cmd, shell=True)
                    
                    with open(f'{LOG1}', 'r') as file:
                        for line in file:
                            match = re.search(r'Total time\(s\):,(\d+\.\d+)', line)
                            if match:
                                bipart = float(match.group(1))
                                break
                    
                    print(f"n={n}, time={bipart}")
                    out.write(str(bipart)+",")
                    out.flush()
                out.write("\n")
            count += 1


# -------------------------------------------------------------------------
# This script runs BiPart saving performance results inside hipeac_results/
# letting the user chose between the best policy for BiPart chosen from the 
# authors file ../results/quality_comp_all_240112.csv or a standard 
# execution without specifying any policy. 
# -------------------------------------------------------------------------
if __name__ == '__main__':
    mode = questionary.select(
        "Select execution mode:",
        choices=[
            "best policy",
            "standard",
        ],
    ).ask()


    start_time = time.time()
    prof_bipart_results(mode)
    elapsed_time = time.time() - start_time  
    print(f"time: {elapsed_time} s, {elapsed_time / 3600} h")
