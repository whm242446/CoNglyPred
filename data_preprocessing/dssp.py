import os
import numpy as np

from tqdm import tqdm

'''
  Getting DSSP files from PDB files
'''

def check_or_create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as error:
            print(f"Error creating directory {folder}: {error}")
            return False
    return True

# PDB folder & DSSP folder
pdb_folder = '/PDB'
output_folder = '/DSSP'


if not check_or_create_folder(pdb_folder) or not check_or_create_folder(output_folder):
    print("Error with input or output directory.")
else:
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    for filename in tqdm(pdb_files, desc="Processing PDB files", unit="file"):
        pdb_path = os.path.join(pdb_folder, filename)
        print(pdb_path)
        output_path = os.path.join(output_folder, filename.replace('.pdb', '.dssp'))

        exit_code = os.system(f'mkdssp -v {pdb_path} {output_path}')  # Using the mkdssp tool version 4.4.0
        if exit_code != 0:
            print(f"Error processing file {filename}. mkdssp exited with code {exit_code}.")


'''
  Extracting structural features from DSSP files
'''

def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature


def main(dssp_folder, output_folder):
    dssp_data = {}
    for filename in os.listdir(dssp_folder):
        if filename.endswith('.dssp'):
            dssp_file = os.path.join(dssp_folder, filename)
            seq, dssp_feature = process_dssp(dssp_file)
            transformed_feature = transform_dssp(dssp_feature)
            filename = os.path.splitext(filename)[0]
            dssp_data[filename] = {'seq': seq, 'dssp_features': transformed_feature}

    np.save(os.path.join(output_folder, 'dssp_features.npy'), dssp_data)

if __name__ == '__main__':
    dssp_folder = '/DSSP'  
    output_folder = ''
    main(dssp_folder, output_folder)
