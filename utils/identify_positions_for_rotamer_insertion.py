import pandas as pd

def AAs_up_to_chi1():
    AAs = ['CYS', 'SER', 'THR', 'VAL']
    return AAs

def AAs_up_to_chi2():
    AAs = ['ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR']
    return AAs

def AAs_up_to_chi3():
    AAs = ['GLN', 'GLU', 'MET']
    return AAs

def AAs_up_to_chi4():
    AAs = ['ARG', 'LYS']
    return AAs

def num_chis_for_residue_id(res_id):
    if res_id in AAs_up_to_chi4():
        return 4
    if res_id in AAs_up_to_chi3():
        return 3
    elif res_id in AAs_up_to_chi2():
        return 2
    elif res_id in AAs_up_to_chi1():
        return 1
    else:
        return 0

def import_fragment_library(library_path:str):
    '''
    reads in a fragment library
    '''
    library = pd.read_pickle(library_path)
    #library.drop(library.columns[[0]], axis=1, inplace=True)
    return library

def check_if_angle_in_bin(df, phi, psi, phi_psi_bin):

    df['phi_difference'] = df.apply(lambda row: angle_difference(row['phi'], phi), axis=1)
    df['psi_difference'] = df.apply(lambda row: angle_difference(row['psi'], psi), axis=1)

    df = df[(df['psi_difference'] < phi_psi_bin / 2) & (df['phi_difference'] < phi_psi_bin / 2)]
    df = df.drop(['phi_difference', 'psi_difference'], axis=1)
    return df


def angle_difference(angle1, angle2):

    return min([abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)])

def main(args):

    row = pd.read_json(args.input_json, lines=True).squeeze()
    rotamer_positions = import_fragment_library(args.fraglib)

    #convert string to list
    if args.rot_sec_struct:
        rot_sec_struct = [*args.rot_sec_struct]
    else:
        rot_sec_struct = None

    #filter based on difference & amino acid identity
    rotamer_positions = rotamer_positions[rotamer_positions['AA'] == row['identity']]
    rotamer_positions = check_if_angle_in_bin(rotamer_positions, row['phi'], row['psi'], args.phi_psi_bin)
    if rot_sec_struct:
        rotamer_positions = rotamer_positions[rotamer_positions['ss'].isin(rot_sec_struct)]
    
    if num_chis_for_residue_id(row['identity']) > 0:
        for chi_angle in range(1, num_chis_for_residue_id(row['identity']) + 1):
            chi = f"chi{chi_angle}"
            rotamer_positions[f'{chi}_difference'] = rotamer_positions.apply(lambda line: angle_difference(line[chi], row[chi]), axis=1)
            rotamer_positions = rotamer_positions[rotamer_positions[f'{chi}_difference'] < row[f'{chi}sig'] * args.chi_std_multiplier]
            rotamer_positions.drop(f'{chi}_difference', axis=1, inplace=True)
            #change chi angles to mean value from bin
            rotamer_positions[chi] = row[chi]
    rotamer_positions['probability'] = row['probability']
    rotamer_positions['phi_psi_occurrence'] = row['phi_psi_occurrence']
    rotamer_positions['rotamer_score'] = row['rotamer_score']
    rotamer_positions['log_occurrence'] = row['log_occurrence']
    rotamer_positions.to_pickle(args.output_pickle)
    
    return 



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="Input rotamer")
    argparser.add_argument("--fraglib", type=str, required=True, help="Path to fragment library")
    argparser.add_argument("--output_pickle", type=str, required=True, help="Path to output pkl file")
    argparser.add_argument("--phi_psi_bin", type=float, default=10, help="Phi-Psi bin size for grouping backbone angles.")
    argparser.add_argument("--rot_sec_struct", type=str, default=None, help="Desired secondary structure at rotamer position.")
    argparser.add_argument("--chi_std_multiplier", type=float, default=1, help="Bin size for filtering fitting chi angles.")

    args = argparser.parse_args()

    main(args)
