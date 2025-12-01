# -------------------------------------------------------------------------
# This is the Tor Vergata team version of run-hmetis.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# For the original version of the code please refer to:
# https://github.com/zlwu92/gHyPart_TACO/blob/master/scripts/run-hmetis.py
# -------------------------------------------------------------------------
import input_header as input
import time
import pandas as pd
import re
import shlex
import datetime

nparts = [2, 3, 4]

current_date = datetime.date.today()

edgecut_output = f"../hipeac_results/hipeac_prof_hmetis_{input.cpu_name}_t{input.num_thread}_{current_date}.csv"

bipart_perf = '../results/bipart_perf_xeon_4214R_24cores_t12.csv'

data = pd.read_csv(bipart_perf)
bp2 = data['k=2']
bp3 = data['k=3']
bp4 = data['k=4']
print(max(bp2), max(bp3), max(bp4))
print(data.iloc[:, 2].max(), data.iloc[:, 3].max(), data.iloc[:, 4].max())

def prof_hmetis_results():
    with open(edgecut_output, 'a') as out:
        count = 490
        for key, value in input.input_map.items():
            if count >= 0:
                out.write(str(count)+","+value+",")
                print(count, value)
                for n in nparts:
                    LOG = "run-hmetis.log"
                    file_path = input.os.path.join(input.dataset_relative_path, key)
                    cmd = f"../cpu_works/hmetis-1.5-linux/hmetis "
                    cmd += f"{file_path} "
                    cmd += f"{n} "  # nparts
                    cmd += f"5 "    # ubfactor
                    cmd += f"5 "   # niter
                    cmd += f"2 "    # CType
                    cmd += f"1 "    # RType
                    cmd += f"3 "    # Vcycle
                    cmd += f"1 "    # ReconstructHE
                    cmd += f"24 "   # Show all information about the multiple runs
                    print(cmd)

                    timeout = data.iloc[:, n].max() * 5
                    print(f"timeout: {timeout}")
                    try:
                        with open(LOG, "w") as log:
                            input.subprocess.run(shlex.split(cmd), stdout=log, stderr=input.subprocess.PIPE)

                        with open(LOG, 'r') as file:
                            text = file.read()
                            
                        pattern1 = r'average = (\d+\.\d+)'
                        average_values = re.findall(pattern1, text)
                        
                        pattern2 = r'Partitioning Time:\s+(\d+\.\d+)sec'
                        match = re.search(pattern2, text)
                        if match:
                            partitioning_time = float(match.group(1))
                            print(f"partitioning_time: {partitioning_time}")
                        print(average_values)
                        total_average = sum(float(value) for value in average_values)
                        
                        match = re.search(r"Hyperedge Cut:\s+(\d+)", text)
                        if match:
                            hyperedge_cut = int(match.group(1))
                            print("Hyperedge Cut =", hyperedge_cut)
                        else:
                            print("Valore non trovato")  
                         
                        print(f"total_average: {total_average}")
                        out.write(f"{hyperedge_cut},{partitioning_time},")

                    except input.subprocess.TimeoutExpired:
                        print("Process timed out")
                        out.write(f"timeout,{timeout},")
                    except Exception as e:
                        if "Signals.SIGSEGV" in str(e):
                            print("Segmentation fault occurred in the external program!")
                            out.write(f"segfault,{timeout},")
                        print("eccezzione")
                    out.flush()
                out.write("\n")
            count += 1
            

# -------------------------------------------------------------------------
# This scripts run and saves hMetis collecting both performance and quality
# results. Note that w.r.t. the authors' work, this script doesn't uses 
# timeout values to stop hMetis computation when is slower than 5x BiPart.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    prof_hmetis_results()
    elapsed_time = time.time() - start_time  
    print(f"time: {elapsed_time} s, {elapsed_time / 3600} h")