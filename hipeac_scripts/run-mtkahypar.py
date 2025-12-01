# -------------------------------------------------------------------------
# This is the Tor Vergata team version of run-mtkahypar.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# For the original version of the code please refer to:
# https://github.com/zlwu92/gHyPart_TACO/blob/master/scripts/run-mtkahypar.py
# -------------------------------------------------------------------------
import input_header as input
import time
import os
import datetime
import csv
import math
import subprocess

nparts = [2, 3, 4]
current_date = datetime.date.today()


edgecut_output = f"../hipeac_results/hipeac_mtkahypar_quality_{input.cpu_name}_t{input.num_thread}_{current_date}.csv"
perf_results = f"../hipeac_results/hipeac_mtkahypar_{input.cpu_name}_t{input.num_thread}_{current_date}.csv"



# -------------------------------------------------------------------------
# Build and run the code. For each run the log is read and the information
# about the performance of the partitions found in terms of execution time 
# is saved as an entry of a .csv file. All the results are saved under 
# hipeac_results/ directory. 
# -------------------------------------------------------------------------
def prof_mtkahypar_results():
    os.system("cd ../cpu_works/Mt-KaHyPar-SDet/ && ./build.sh ")
    
    with open(perf_results, 'a') as out:
        count = 0
        for key, value in input.input_map.items():
            file_path = os.path.join(input.dataset_relative_path, key)
            if count >= 0:
                out.write(str(count)+","+value+",")
                print(count, value)
                for n in nparts:
                    LOG1 = "run-mtkahypar.log"
                    output_folder = f"out/mtkahypar/{current_date}"
                    print(output_folder)
                    cmd = f"../cpu_works/Mt-KaHyPar-SDet/build/mt-kahypar/application/MtKaHyPar -h "
                    cmd += f"{file_path} "
                    cmd += f"-p ../cpu_works/Mt-KaHyPar-SDet/config/deterministic_preset.ini "
                    cmd += f"--instance-type=hypergraph "
                    cmd += f"-k {n} "
                    cmd += f"-e 0.05 -o cut -m direct -t 12 --write-partition-file=true --partition-output-folder={output_folder} > {LOG1}"
                    print(cmd)
                    subprocess.call(cmd, shell=True)

                    mtkahypar = []
                    with open(LOG1, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            if row and ('+ Preprocessing' in row[0] or '+ Coarsening' in row[0] or '+ Initial' in row[0] or '+ Refinement' in row[0] or '+ Postprocessing' in row[0]):
                                substr = row[0][48:]
                                result = substr[:len(substr)-2]
                                mtkahypar.append(float(result))
                    result2 = '{:.3f}'.format(math.fsum(mtkahypar))
                    print(f"n={n}, time={result2}")
                    out.write(str(result2)+",")
                    out.flush()
                out.write("\n")
            count += 1

# -------------------------------------------------------------------------
# Build and run the code. For each run the log is read and the information
# about the quality of the partitions found in terms of edge cut cardinality
# is saved as an entry of a .csv file. All the results are saved under 
# hipeac_results/ directory 
# -------------------------------------------------------------------------
def prof_mtkahypar_quality_results():
    os.system("cd ../cpu_works/Mt-KaHyPar-SDet/ && ./build.sh ")
    
    with open(edgecut_output, 'w') as out:
        out.write("id,dataset,k=2,k=3,k=4\n")
        
    with open(edgecut_output, 'a') as out:
        count = 0
        for key, value in input.input_map.items():
            file_path = os.path.join(input.dataset_relative_path, key)
            if count >= 0:
                out.write(str(count)+","+value+",")
                print(count, value)
                for n in nparts:
                    LOG1 = "run-mtkahypar.log"
                    cmd = f"../cpu_works/Mt-KaHyPar-SDet/build/mt-kahypar/application/MtKaHyPar -h "
                    cmd += f"{file_path} "
                    cmd += f"-p ../cpu_works/Mt-KaHyPar-SDet/config/deterministic_preset.ini "
                    cmd += f"--instance-type=hypergraph "
                    cmd += f"-k {n} "
                    cmd += f"-e 0.05 -o cut -m rb -t 12 > {LOG1}"
                    print(cmd)
                    os.system(cmd)

                    time = []
                    cut = []
                    with open(LOG1, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            if row and ('+ Preprocessing' in row[0] or '+ Coarsening' in row[0] or '+ Initial' in row[0] or '+ Refinement' in row[0] or '+ Postprocessing' in row[0]):
                                substr = row[0][48:]
                                result = substr[:len(substr)-2]
                                time.append(float(result))
                            if row and "Hyperedge Cut" in row[0]:
                                cut.append(int(row[0][29:]))
                                
                    result2 = '{:.3f}'.format(math.fsum(time))
                    print(f"n={n}, cut={cut}, time={result2}")
                    out.write(f"{cut[0]},")
                    out.flush()
                out.write("\n")
            count += 1


# -------------------------------------------------------------------------
# This main calls mt-khypar exclusively to get performance results in term 
# of execution time or qualiuty results in term of edge cut cardinality.
#
# THe script should be updated in order to collect all the data in a single 
# execution in order to avoid hours of computational time since information
# about performance and quality are both present in each log after a single 
# run, so there is no real need to re-run the same code twice.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    # prof_mtkahypar_results()
    prof_mtkahypar_quality_results()
    elapsed_time = time.time() - start_time  
    print(f"time: {elapsed_time} s, {elapsed_time / 3600} h")
