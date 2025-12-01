# -------------------------------------------------------------------------
# This is the Tor Vergata team version of run_all_pattern.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# For the original version of the code please refer to:
# https://github.com/zlwu92/gHyPart_TACO/blob/master/scripts/run_all_pattern.py
# -------------------------------------------------------------------------
import input_header as input
import time
from patterns import all_patterns as ap
import datetime
import os

current_date = datetime.date.today()

outfile = f'../hipeac_results/hipeac_benchmarks_all_patterns_real_hypergraphs_{input.gpu_name}_t{input.num_thread}_{current_date}.csv'

def brute_force_each_pattern():
    os.system(f"mkdir -p ../build && cd ../build && cmake .. && make -j8 ")
    
    print("Savikng results on ", outfile)

    with open(outfile, 'a') as out:
        out.write("id,dataset,")
        out.write("p1_k1235_p1_k4k6_p2_k12,p2_k1235_p1_k4k6_p2_k12,p3_k1235_p1_k4k6_p2_k12,")
        out.write("p1_k1235_p2_k4k6_p2_k12,p2_k1235_p2_k4k6_p2_k12,p3_k1235_p2_k4k6_p2_k12,")
        out.write("p1_k1235_p3_k4k6_p2_k12,p2_k1235_p3_k4k6_p2_k12,p3_k1235_p3_k4k6_p2_k12,")
        out.write("p1_k1235_p1_k4k6_p3_k12,p2_k1235_p1_k4k6_p3_k12,p3_k1235_p1_k4k6_p3_k12,")
        out.write("p1_k1235_p2_k4k6_p3_k12,p2_k1235_p2_k4k6_p3_k12,p3_k1235_p2_k4k6_p3_k12,")
        out.write("p1_k1235_p3_k4k6_p3_k12,p2_k1235_p3_k4k6_p3_k12,p3_k1235_p3_k4k6_p3_k12,")
        out.write("p1_k1235_p1_k4k6_p2b_k12,p2_k1235_p1_k4k6_p2b_k12,p3_k1235_p1_k4k6_p2b_k12,")
        out.write("p1_k1235_p2_k4k6_p2b_k12,p2_k1235_p2_k4k6_p2b_k12,p3_k1235_p2_k4k6_p2b_k12,")
        out.write("p1_k1235_p3_k4k6_p2b_k12,p2_k1235_p3_k4k6_p2b_k12,p3_k1235_p3_k4k6_p2b_k12,")
        out.write("gHyPart,best,gHyPart_speedup,best_speedup,")
        out.write("matching_select_pattern,matching_best_pattern,")
        out.write("node_merging_select_pattern,node_merging_best_pattern,")
        out.write("construction_select_pattern,construction_best_pattern,")
        out.write("avghedgesize,sdv/avg,edge_cut\n")
        count = 0
        for key, value in input.input_map.items():
            if count >= 0:
                print(str(count), value)
                out.write(str(count)+","+value+",")
                file_path = input.os.path.join(input.dataset_relative_path, key)
                our_impl = []
                patterns = {}
                best = 0.0

                LOG = "../results/ours.log"

                ap.p1_k1235_p1_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p1_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p1_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p2_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p2_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p2_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p3_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p3_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p3_k4k6_p2_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p1_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p1_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p1_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p2_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p2_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p2_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p3_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p3_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p3_k4k6_p3_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p1_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p1_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p1_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p2_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p2_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p2_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                
                ap.p1_k1235_p3_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p2_k1235_p3_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                ap.p3_k1235_p3_k4k6_p2b_k12(out, file_path, our_impl, patterns, LOG)
                
                best = min(our_impl)
                print("cur_best:" + str(best))
                # '''
                selection_k12_pattern = ''
                selection_k4_pattern = ''
                selection_k1_pattern = ''
                avghedgesize = ''
                relative_sdv = ''
                select_pattern = {}
                edge_cut = ''
                best_k12_pattern = ''
                best_k4_pattern = ''
                best_k1_pattern = ''
                best_speedup = 0.0
                select_speedup = 0.0
                ap.select_pattern(out, file_path, our_impl, best, 
                                selection_k12_pattern, selection_k4_pattern, selection_k1_pattern,
                                avghedgesize, relative_sdv, select_pattern, patterns, edge_cut, 
                                best_k12_pattern, best_k4_pattern, best_k1_pattern,
                                best_speedup, select_speedup, LOG)
            count+=1    

# -------------------------------------------------------------------------
# This script executes ghypart-O in a brute force approach trying all the
# possible optimizations and keeping the best one. Minimal changes were 
# done from the authors' version of the script.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    brute_force_each_pattern()
    elapsed_time = time.time() - start_time  
    print(f"time: {elapsed_time} s, {elapsed_time / 3600} h")
