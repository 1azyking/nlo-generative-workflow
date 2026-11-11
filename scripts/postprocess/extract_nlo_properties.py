import os
import re
import csv
import argparse


def extract_data(
    work_dir,
    csv_path
):
    if os.path.isdir(csv_path):
        output_csv = os.path.join(csv_path, "output.csv")
    elif os.path.isfile(csv_path) and csv_path.endswith(".csv"):
        output_csv = csv_path
    else:
        raise ValueError(f"Invalid csv path: {csv_path}, expected either a directory or a csv file")
    results = []
    for compound_folder in os.listdir(work_dir):
        compound_path = os.path.join(work_dir, compound_folder)
        if not os.path.isdir(compound_path):
            continue
        nlo_log_path = os.path.join(compound_path, "nlo.log")
        if os.path.exists(nlo_log_path):
            dij_data = {}
            with open(nlo_log_path, "r") as log_file:
                lines = log_file.readlines()
                for i, line in enumerate(lines):
                    if "d(" in line and "SHG (real part)=" in line:
                        match = re.search(
                            r"d\(\s*(\d),\s*(\d)\).*?SHG \(real part\)=\s*([-+]?\d*\.\d+|\d+)",
                            line
                        )
                        if match:
                            ij = (int(match.group(1)), int(match.group(2)))
                            real_part = float(match.group(3))
                            dij_data[ij] = abs(real_part)
            if dij_data:
                max_real_value = max(dij_data.values())
                max_ij = [ij for ij, value in dij_data.items() if value == max_real_value]

        birefringence_dir = next((d for d in os.listdir(compound_path) if d.startswith("lo_")), None)
        birefringence_path = os.path.join(compound_path, birefringence_dir,
                                          "birefringence.dat") if birefringence_dir else None
        dn_1064 = None
        dn_546 = None
        if birefringence_path and os.path.exists(birefringence_path):
            with open(birefringence_path, "r") as bire_file:
                for line in bire_file:
                    match = re.search(
                        r"([\d\.E+-]+)\s+([\d\.E+-]+)",
                        line
                    )
                    if match:
                        E = float(match.group(1))
                        dn = float(match.group(2))
                        if abs(E - 1.165) < 1e-4:
                            dn_1064 = dn
                        elif abs(E - 2.270) < 1e-4:
                            dn_546 = dn

        if dij_data:
            results.append([
                compound_folder,
                max_real_value,
                ";".join([f"({ij[0]},{ij[1]})" for ij in max_ij]),
                dn_1064 if dn_1064 is not None else "N/A",
                dn_546 if dn_546 is not None else "N/A"
            ])

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Compound", "dij_value", "ij_pairs", "dn_1064nm", "dn_546nm"])
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract NLO properties from out files",
    )
    parser.add_argument("work_dir", help="Directory of out files", )
    parser.add_argument("csv_path", help="Path of csv file to store nlo props")
    args = parser.parse_args()
    extract_data(args.work_dir, args.csv_path)
