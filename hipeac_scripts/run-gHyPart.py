# -------------------------------------------------------------------------
# This is the Tor Vergata team version of run-gHyPart.py used for the 
# reproducibility study of gHyPart for HiPEAC Students Challenge 2025.
#
# For the original version of the code please refer to:
# https://github.com/zlwu92/gHyPart_TACO/blob/master/scripts/run-gHyPart.py
# -------------------------------------------------------------------------
import input_header as input
import time
import pandas as pd
import os
import re
import datetime
import questionary

current_date = datetime.date.today()
nparts = [2, 3, 4]        # k values
log_file = "run-ghypart.log"

# Read reference policies (required by gHyPart)
ref_results = "../results/quality_comp_all_240112.csv"
gp_policy = pd.read_csv(ref_results)['policy1']

# Output CSV files
ghypart_b_outfile = f'../hipeac_results/hipeac_ghypart_B_all_{input.gpu_name}_t{input.num_thread}_{current_date}.csv'
ghypart_outfile   = f'../hipeac_results/hipeac_ghypart_all_{input.gpu_name}_t{input.num_thread}_{current_date}.csv'
ghypart_b_best_outfile = f'../hipeac_results/hipeac_ghypart_B_all_best_{input.gpu_name}_t{input.num_thread}_{current_date}.csv'
ghypart_best_outfile   = f'../hipeac_results/hipeac_ghypart_all_best_{input.gpu_name}_t{input.num_thread}_{current_date}.csv'


# ----------------------------------------------------------------------
# Build project
# ----------------------------------------------------------------------
def build_project():
    """Compile and create directories."""
    cmds = [
        "mkdir -p ../build && cd ../build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make -j",
        "mkdir -p out",
        "cd out && mkdir -p ghypart",
        f"cd out/ghypart && mkdir -p {current_date}",
    ]
    for c in cmds:
        os.system(c)


# ----------------------------------------------------------------------
# Parsers (return last match found)
# ----------------------------------------------------------------------
def parse_quality(log_file):
    """Return last '# of Hyperedge cut:' value."""
    last_val = None
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"# of Hyperedge cut:\s*(\d+)", line)
            if m:
                last_val = int(m.group(1))
    return last_val if last_val is not None else 0


def parse_perf(log_file):
    """Return last 'Total k-way partition time' value."""
    last_val = None
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"Total k-way partition time \(s\):\s*([\d.]+)", line)
            if m:
                last_val = float(m.group(1))
    return last_val if last_val is not None else 0.0


# ----------------------------------------------------------------------
# Main experiment loop
# ----------------------------------------------------------------------
def run_experiment(flags, module, mode):
    """Run gHyPart for all datasets, storing both perf & quality for each k."""

    # Select the correct output file
    match (module, mode):
        case ("ghypart", "best policy"):
            outfile = ghypart_best_outfile
        case ("ghypart", "standard"):
            outfile = ghypart_outfile
        case ("ghypart_b", "best policy"):
            outfile = ghypart_b_best_outfile
        case ("ghypart_b", "standard"):
            outfile = ghypart_b_outfile
        case _:
            raise ValueError("Invalid module/mode combination")

    print("Saving results on ", outfile)


    # CSV header
    with open(outfile, "w") as out:
        out.write(
            "id,dataset,hedges,"
            "k=2_cut,k=3_cut,k=4_cut,"
            "k=2_perf,k=3_perf,k=4_perf\n"
        )

    # Append entries
    with open(outfile, "a") as out:
        for idx, (filename, dataset_name) in enumerate(input.input_map.items()):

            print(f"\n--- Dataset {idx}: {dataset_name} ---")

            file_path = input.os.path.join(input.dataset_relative_path, filename)

            # Read total_hyperedges from first line of .hgr
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
            total_hyperedges = int(first_line.split()[0])

            # Prepare per-k results
            cuts = []
            perfs = []

            # Select policy based on execution mode
            if mode == "best policy":
                policy_value = gp_policy[idx]
            else:
                policy_value = ""   # default / standard behavior

            # Run gHyPart for each k
            for n in nparts:
                cmd = (
                    f"../build/gHyPart {file_path} {flags} "
                    f"{policy_value} -numPartitions {n} > {log_file} 2>&1"
                )

                print(cmd)
                input.subprocess.call(cmd, shell=True)

                # Collect results
                cut = parse_quality(log_file)
                perf = parse_perf(log_file)

                cuts.append(cut)
                perfs.append(perf)

                print(f"k={n}: cut={cut}, perf={perf}")

            # Write row
            out.write(
                f"{idx},{dataset_name},{total_hyperedges},"
                f"{cuts[0]},{cuts[1]},{cuts[2]},"
                f"{perfs[0]},{perfs[1]},{perfs[2]}\n"
            )


# ----------------------------------------------------------------------
# Available flag sets
# ----------------------------------------------------------------------
flags_dict = {
    "ghypart": "-bp -wc 0 -useuvm 0 -sort_input 0 -useSelection 1 -useMemOpt 1",
    "ghypart_b": "-bp -wc 0 -useuvm 0 -sort_input 0 -usenewkernel12 0 -newParBaseK12 1 -useMemOpt 1",
}


# ----------------------------------------------------------------------
# This scripts lets the user run ghypart and ghypart-B. In both cases 
# the scripts saves results under hipeac_results/ and saves both 
# performance and quality information. Also this scripts runs ghypart
# once for each hypergraph in the datasets and reads the output log in 
# order to collect information each time.
#
# This scripts runs also ghypart and ghypart-B using the best policies 
# for quality results. These policies were found inside authors' result 
# file  "../results/quality_comp_all_240112.csv" file and there is no 
# other documented way to reobtain these best policies results.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    module = questionary.select(
        "Which module do you want to run?",
        choices=["ghypart", "ghypart_b"],
    ).ask()

    mode = questionary.select(
        "Select execution mode:",
        choices=[
            "best policy",
            "standard",
        ],
    ).ask()

    build_project()

    print(f"\nRunning {module} (full quality + performance)...")

    start = time.time()
    run_experiment(flags_dict[module], module, mode)
    elapsed = time.time() - start

    print(f"\nDone! Total time: {elapsed:.3f} s ({elapsed/3600:.3f} h)\n")