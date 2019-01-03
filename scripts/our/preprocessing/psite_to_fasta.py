import argparse
import os
from pathlib import Path
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Parse psite files into fasta files')
parser.add_argument('-d', '--pisite_directory')
parser.add_argument('-o', '--output_file')
args = parser.parse_args()


def process_file(file_name):
    with open(file_name, 'r') as file:

        name = ''
        sequence = ''
        ppi = ''
        skip = True
        for line in file:
            if skip:
                if line.startswith('#PDBID'):
                    name = line.split()[1]
                elif line.startswith('#CHAIN'):
                    name += line.split()[1]
                elif line.startswith('#residue'):
                    skip = False
            else:
                content = line.split()
                if len(content) < 3:
                    break

                sequence += content[1]
                ppi += '+' if int(content[2]) > 0 else '-'
    return f">{name}\n{sequence}\n{ppi}\n"

# get list of pisite files
file_names = list(Path(args.pisite_directory).rglob("*.pisite"))
print(f"{len(file_names)} pisite files")

# read files in parallel
print("read files")
with Pool(processes=8) as pool:
    map = pool.map(process_file, file_names)

# open fasta output file)
print("write files")
with open(args.output_file, 'w') as file:
    file.writelines(map)
