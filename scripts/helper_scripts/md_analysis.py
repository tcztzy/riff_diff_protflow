# For MD simulation analysis this script executes the following 
# MD simulation analysis:
# 1. After equilibration check RMSD to ideal geometry. Large shift should indicated a “bad” model.
# 2. During MD sampling look at the chi angle changes, this should indicate conformational stability.
# 3. Figure out a way to measure if there are consorted motions and differences between design’s motions.
# 4. Measure CA and all atom RMSD change.
#
# In order to do so, we first must match the ideal geometry to design

# %%
import MDAnalysis as mda
import pandas as pd
import numpy as np
import os
from typing import NamedTuple
from scipy.spatial.distance import cdist, pdist
from munkres import Munkres
import functools
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis import base as mda_base
from io import StringIO
from itertools import combinations
from MDAnalysis.lib.distances import calc_dihedrals
import fire
from pathlib import Path
#import propkatraj
from MDAnalysis.analysis.distances import dist

import warnings
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')

# %%
CHI_ANGLES='''
angle, aa, s1, s2, s3, s4
CHI1, ARG, N, CA, CB, CG 
CHI1, ASN, N, CA, CB, CG 
CHI1, ASP, N, CA, CB, CG 
CHI1, CYS, N, CA, CB, SG 
CHI1, GLN, N, CA, CB, CG 
CHI1, GLU, N, CA, CB, CG 
CHI1, HIS, N, CA, CB, CG 
CHI1, ILE, N, CA, CB, CG1
CHI1, LEU, N, CA, CB, CG 
CHI1, LYS, N, CA, CB, CG 
CHI1, MET, N, CA, CB, CG 
CHI1, PHE, N, CA, CB, CG 
CHI1, PRO, N, CA, CB, CG 
CHI1, SER, N, CA, CB, OG 
CHI1, THR, N, CA, CB, OG1
CHI1, TRP, N, CA, CB, CG 
CHI1, TYR, N, CA, CB, CG 
CHI1, VAL, N, CA, CB, CG1
CHI2, ARG, CA, CB, CG, CD 
CHI2, ASN, CA, CB, CG, OD1
CHI2, ASP, CA, CB, CG, OD1
CHI2, GLN, CA, CB, CG, CD 
CHI2, GLU, CA, CB, CG, CD 
CHI2, HIS, CA, CB, CG, ND1
CHI2, ILE, CA, CB, CG1, CD
CHI2, LEU, CA, CB, CG, CD1
CHI2, LYS, CA, CB, CG, CD 
CHI2, MET, CA, CB, CG, SD 
CHI2, PHE, CA, CB, CG, CD1
CHI2, PRO, CA, CB, CG, CD 
CHI2, TRP, CA, CB, CG, CD1
CHI2, TYR, CA, CB, CG, CD1
CHI3, ARG, CB, CG, CD, NE 
CHI3, GLN, CB, CG, CD, OE1
CHI3, GLU, CB, CG, CD, OE1
CHI3, LYS, CB, CG, CD, CE 
CHI3, MET, CB, CG, SD, CE 
CHI4, ARG, CG, CD, NE, CZ
CHI4, LYS, CG, CD, CE, NZ
CHI5, ARG, CD, NE, CZ, NH1
'''

class ResNumCAMap(NamedTuple):
   design_to_ideal: dict[int, int]
   ideal_to_design: dict[int, int]

class ResIdxCAMap(NamedTuple):
   design_to_ideal: dict[int, int]
   ideal_to_design: dict[int, int]

class CAMap(NamedTuple):
    resnum_map: ResNumCAMap
    residx_map: ResIdxCAMap

class Janin(mda_base.AnalysisBase):
    
    def __init__(self, atomgroup, check_protein=True, verbose=False, **kwargs):
       super().__init__(atomgroup.universe.trajectory, verbose, **kwargs)
       self.atomgroup = atomgroup
       residues = self.atomgroup.residues
       resnums = residues.resnums

       self.aa_chi_df = pd.read_csv(StringIO(CHI_ANGLES.strip()), skipinitialspace=True).rename(columns = lambda x: x.strip())
       self.aa_chi_df = self.aa_chi_df.map(lambda s: s.strip())
       self.chi_atom_selectors = self.aa_chi_df.groupby('angle')[['s1', 's2', 's3', 's4']].apply(lambda x: ('name ' + ' '.join(x['s1'].unique()), 'name ' + ' '.join(x['s2'].unique()), 'name ' + ' '.join(x['s3'].unique()), 'name ' + ' '.join(x['s4'].unique()))).to_dict()
       self.chi_aa_selectors = 'resname ' + self.aa_chi_df.groupby('angle')['aa'].apply(lambda x: ' '.join(x))

       self.results.angles = []
    #    chi_df = pd.DataFrame({'resnum': [], 'chi': [], 'angle': []}) # {'resnum': resnums}).set_index('resnum')
       self.chi_ag = {}
       for chi_angle in self.aa_chi_df['angle'].unique():
        #    self.results.chi_df[chi_angle] = pd.NA
           self.chi_ag[chi_angle] = [self.atomgroup.select_atoms(self.chi_aa_selectors[chi_angle] + ' and ' + atom_selector) for atom_selector in self.chi_atom_selectors[chi_angle]]


       if check_protein:
            protein = self.atomgroup.universe.select_atoms("protein").residues

            if not residues.issubset(protein):
                raise ValueError("Found atoms outside of protein. Only atoms "
                                "inside of a 'protein' selection can be used to "
                                "calculate dihedrals.")
    
    # def _prepare(self):
    #     self.results.chi_df[:] = pd.NA
        
    def _single_frame(self):
        for chi_angle, ags in self.chi_ag.items():
            angles = calc_dihedrals(*ags, box=ags[0].dimensions)
            resnums = ags[0].resnums
            chi = np.array([chi_angle] * len(resnums))
            ts = np.array([self._ts.time] * len(resnums))
            self.results.angles.append(pd.DataFrame({
                'resnum': resnums,
                'chi': chi,
                'angle': angles,
                'ts': ts
            }))
            # self.results.chi_df.loc[resnums, chi_angle] = angles

    def _conclude(self):
        self.results.chi_df = pd.concat(self.results.angles).reset_index()

def sort_atoms_within_residues(atomgroup):
    # get all allowed atom indeces
    atom_indices_set = set(atomgroup.indices)

    # sort atoms within each residue by atom name
    sorted_atoms = []
    for residue in atomgroup.residues:
        sorted_atoms.extend(sorted([atom for atom in residue.atoms if atom.index in atom_indices_set], key=lambda atom: atom.name))

    # extract indices of sorted atoms
    sorted_atom_indeces = [atom.index for atom in sorted_atoms]
    return atomgroup.universe.atoms[sorted_atom_indeces]
            
class InternalDistances(mda_base.AnalysisBase):

    def __init__(self, atomgroup, reference_atomgroup=None, verbose=False, **kwargs):
        super().__init__(atomgroup.universe.trajectory, verbose, **kwargs)
        self.atom_group = atomgroup
        if reference_atomgroup is not None:
            assert len(reference_atomgroup) == len(atomgroup)
            self.reference_pdist = pdist(reference_atomgroup.positions)
        else:
            self.reference_pdist = pdist(atomgroup.positions)
    
    def _prepare(self):
        self.results.internal_distances = []
        self.results.internal_distances_deviation = []
    
    def _single_frame(self):
        internal_dist = pdist(self.atom_group.positions)
        self.results.internal_distances.append(internal_dist)
        self.results.internal_distances_deviation.append(np.linalg.norm(internal_dist - self.reference_pdist)/len(self.atom_group))

class ReferenceDistances(mda_base.AnalysisBase):
    def __init__(self, atomgroup, reference_atomgroup=None, verbose=False, **kwargs):
        super().__init__(atomgroup.universe.trajectory, verbose, **kwargs)
        self.atom_group = atomgroup
        self.reference_atomgroup = reference_atomgroup
    
    def _prepare(self):
        self.results.reference_distances = []
    
    def _single_frame(self):
        reference_dist = dist(self.atom_group, self.reference_atomgroup)
        self.results.reference_distances.append(reference_dist)

def kabsch_umeyama(A: np.array, B: np.array) -> tuple[np.array, float, np.array]:
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

def rotation_translation_transform(R: np.array, c: float, t: np.array, mat: np.array ) -> np.array:
    # Transform a n,3 array by a rotation and translation
    # c * np.einsum('ij,jk->ik', R, test).T + t
    return c * np.einsum('ij,kj->ik', R, mat).T + t

def catalytic_positions_ca_selection(catalytic_positions: list[int]) -> str:
    return 'name CA and (' + ' or '.join(map(lambda x: f'resid {x}', catalytic_positions)) + ')'

def catalytic_positions_selection(catalytic_positions: list[int]) -> str:
    return '(' + ' or '.join(map(lambda x: f'resid {x}', catalytic_positions)) + ') and not type H'

def get_design_catlytic_ca(u: mda.Universe, catalytic_positions_resnum: list[str]) -> np.array:
    ag = u.atoms[[]]
    for resnum in catalytic_positions_resnum:
        ag += u.select_atoms(f'resid {resnum} and name CA')
    return ag.positions

def get_ideal_geometry_ca(df: pd.DataFrame, catalytic_positions: dict[str, list[int]]) -> np.array:
    catalytic_positions_masks = [
        np.logical_and(df.chain_id == chain_id, df.residue_number.isin(residue_numbers))
        for chain_id, residue_numbers in catalytic_positions.items()
    ]
    catalytic_positions_mask = functools.reduce(np.logical_or, catalytic_positions_masks)
    mask = functools.reduce(np.logical_and, [catalytic_positions_mask, df.atom_name == 'CA'])
    return df.loc[mask, ['x_coord', 'y_coord', 'z_coord']].values

def make_sure_single_min_distances(indices: list[tuple[int, int]]) -> bool:
    x_indices = (idx for idx, _ in indices)
    y_indices = (idx for _, idx in indices)
    return len(set(x_indices)) == len(indices) and len(set(y_indices)) == len(indices)

def find_closest_point(design_ca: np.array, idealg_ca: np.array) -> tuple[dict[int, int], dict[int, int]]:
    if design_ca.shape != idealg_ca.shape:
        raise ValueError(f'Given arrays differ in shape, check inputs! design shape {design_ca.shape} ideal shape {idealg_ca.shape}')

    distances = cdist( design_ca - design_ca.mean(axis=0), idealg_ca - idealg_ca.mean(axis=0))
    m = Munkres()
    indices = m.compute(distances.tolist())
    if not make_sure_single_min_distances(indices):
        raise ValueError(f'Unable to find a single closest point one-to-one mapping. At least one CA has minimal distance to more than one other CA.')
    design_to_ideal_map = dict((idx, val) for idx, val in indices)
    ideal_to_design_map = dict((val, idx) for idx, val in indices)
    
    return design_to_ideal_map, ideal_to_design_map

def transform_map_to_dict(map: dict[int, int], index_names: list[int], value_names: list[int]) -> dict[int, int]:
    return dict((index_names[key], value_names[val]) for key, val in map.items())

def match_ideal_geometry(u: mda.Universe, catalytic_positions: dict[str, list[int]], ideal_geometry: pd.DataFrame):
    # This includes several steps:
    # 1. Extract active site residues' CA coordinates
    #   for both design and ideal geometry (idealG)
    # 2. Find closest CAs between design and idealG.
    # 3. Return a mapping of residue numbers to residue numbers.

    # Throws: ValueError:
    # if there is no clear one to one mapping between desgin and idealG
    design_resnums = [resnum for _, resnums in catalytic_positions.items() for resnum in resnums]
    ideal_geometry_resnums = design_resnums.copy()
    design_cas = get_design_catlytic_ca(u, design_resnums)
    ideal_geometry_cas = get_ideal_geometry_ca(ideal_geometry, catalytic_positions)
    closest_points = find_closest_point(design_cas, ideal_geometry_cas)

    design_to_ideal_dict = transform_map_to_dict(closest_points[0], design_resnums, ideal_geometry_resnums)
    ideal_to_design_dict = transform_map_to_dict(closest_points[1], ideal_geometry_resnums, design_resnums)
    num_map = ResNumCAMap(design_to_ideal=design_to_ideal_dict, ideal_to_design=ideal_to_design_dict)
    idx_map = ResIdxCAMap(design_to_ideal=closest_points[0], ideal_to_design=closest_points[1])
    return CAMap(resnum_map=num_map, residx_map = idx_map)

def design_to_ideal_kabasch(u: mda.Universe, catalytic_positions: dict[str, list[int]], ideal_geometry: pd.DataFrame, design_to_ideal_map: dict[int, int]) -> tuple[np.array, float, np.array]:
    design_resnums = [resnum for _, resnums in catalytic_positions.items() for resnum in resnums]
    design_cas = get_design_catlytic_ca(u, design_resnums)
    ideal_geometry_cas = get_ideal_geometry_ca(ideal_geometry, catalytic_positions)
    ideal_row_ordering = list(dict(sorted(design_to_ideal_map.items())).values())
    reordered_ideal_cas = ideal_geometry_cas[ideal_row_ordering, :]
    return kabsch_umeyama(design_cas, reordered_ideal_cas)
    
# %%
def main(gro_file:str, trajectory_file:str, reference_frag: str, catalytic_positions: list[int], output_dir: str):
    """Run analysis on MD simulation

    Args:
        gro_file (str): Path to gro file
        trajectory_file (str): Path to trajectory_file
        reference_frag (str): Path to PDB reference fragments
        catalytic_positions (list[int]): List of the catalytic positions
        output_path (str): The path to which output DataFrames will be written

    """
    # get pose name by extracting basename and removing "_t0.gro"
    description = os.path.basename(gro_file).replace("_t0.gro", "")
    pose_dir = os.path.join(output_dir, description)
    output_path = f"{pose_dir}/{description}"

    # universe = mda.Universe('/Users/sh/Downloads/md_fix/noHOH.gro', '/Users/sh/Downloads/md_fix/traj_noHOH.xtc')
    universe = mda.Universe(gro_file, trajectory_file)
    ideal_geometry = mda.Universe(reference_frag)

    # create atomgroup selections for analysis
    traj_catres = sort_atoms_within_residues(universe.select_atoms(catalytic_positions_selection(catalytic_positions)))
    ideal_catres = sort_atoms_within_residues(ideal_geometry.select_atoms(catalytic_positions_selection(catalytic_positions)))

    not_sorted_traj = universe.select_atoms(catalytic_positions_selection(catalytic_positions)).names
    not_sorted_ideal = ideal_geometry.select_atoms(catalytic_positions_selection(catalytic_positions)).names

    #if not_sorted_traj != not_sorted_ideal:
    #    print(f"Atom ordering of catalytic residues in trajectory and ideal geometry is not the same!")
    #    if traj_catres.names != ideal_catres.names:
    #        raise ValueError(f"Atom ordering in sorted atom lists is also not the same.\ntraj: {traj_catres.names}\nideal: {ideal_catres}")
    print(not_sorted_traj)
    print(not_sorted_ideal)
    print(traj_catres.names)
    print(ideal_catres.names)
    
    rmsd_analysis = rms.RMSD(
        universe,
        reference=ideal_geometry,
        select=catalytic_positions_ca_selection(catalytic_positions),
        groupselections=[catalytic_positions_selection(catalytic_positions)]
    )

    rmsd_analysis.run()
    rmsd_df = pd.DataFrame(rmsd_analysis.results.rmsd[:, 2:],
                           columns=['C-alphas', 'Protein'],
                           index=rmsd_analysis.results.rmsd[:, 1])
    rmsd_df.index.name = 'Time (ps)'
    rmsd_df.to_csv((rmsd_df_path := output_path.rstrip("/") + '_rmsd.df.csv'))

    # ideal_geometry = cpdb.parse(, df=True)
    # cmap = match_ideal_geometry(universe, catalytic_positions, ideal_geometry)
    # R, c, t = design_to_ideal_kabasch(universe, catalytic_positions, ideal_geometry, cmap.residx_map.design_to_ideal)

    ca_internal_distances = InternalDistances(traj_catres, reference_atomgroup=ideal_catres)
    ca_internal_distances.run()
    ca_internal_df = pd.DataFrame(
        ca_internal_distances.results.internal_distances,
        columns=list(
            map('-'.join,
                combinations(
                    ca_internal_distances.atom_group.resnums.astype(str) + ca_internal_distances.atom_group.names,
                    2
                )
            )
        )
    )
    ca_internal_df.to_csv((ca_internal_df_path := output_path.rstrip("/") + '_ca.internal.df.csv'))
    ref_ca_internal_df = pd.DataFrame( ca_internal_distances.reference_pdist.reshape(1, -1))
    ref_ca_internal_df.columns = ca_internal_df.columns
    ref_ca_internal_df.to_csv((ca_internal_ref_df_path := output_path.rstrip("/") + '_ca.internal.ref.df.csv'))

    internal_distances = InternalDistances(traj_catres, reference_atomgroup=ideal_catres)
    internal_distances.run()
    internal_df = pd.DataFrame(
        internal_distances.results.internal_distances,
        columns=list(
            map('-'.join,
                combinations(
                    internal_distances.atom_group.resnums.astype(str) + internal_distances.atom_group.names,
                    2
                )
            )
        )
    )
    internal_df.to_csv((internal_df_path := output_path.rstrip("/") + '_internal.df.csv'))
    ref_internal_df = pd.DataFrame( internal_distances.reference_pdist.reshape(1, -1) )
    ref_internal_df.columns = internal_df.columns
    ref_internal_df.to_csv((internal_ref_df_path := output_path.rstrip("/") + '_internal.ref.df.csv'))

    janin = Janin(universe.select_atoms(catalytic_positions_selection(catalytic_positions)))
    janin.run()
    chi_df = pd.concat(janin.results.angles)
    chi_df.to_csv((chi_df_path := output_path.rstrip("/") + '_chi.df.csv'))

    #pkatraj = propkatraj.PropkaTraj(universe, skip_failure=True)
    #pkatraj.run()
    #pkatraj.results.pkas.to_csvoutput_path.rstrip("/") + '_pkas.df.csv')

    # align trajectories along Ca of reference geometry
    aligner = align.AlignTraj(universe, ideal_geometry, select='protein and ' + catalytic_positions_ca_selection(catalytic_positions), in_memory=True)
    aligner.run()

    # calc distances between all atoms in catalytic positions to ideal geometry.
    refdist = ReferenceDistances(traj_catres, ideal_catres).run()
    results = np.array(refdist.results.reference_distances) # shape (5001, 3, 41)

    # collect and store results
    results_df = pd.DataFrame(
        results[:, 2, :],
        columns = traj_catres.names + "-" + traj_catres.resnums.astype(str)
    )
    results_df.to_csv((reference_distances_df_path := output_path.rstrip("/") + '_reference_distances.df.csv'))

    rmsf = rms.RMSF(traj_catres).run()
    rmsf_resname = [atom.resname for atom in traj_catres]
    rmsf_resnum = [atom.resnum for atom in traj_catres]
    rmsf_atomname = [atom.name for atom in traj_catres]
    data = np.vstack([rmsf_resname, rmsf_resnum, rmsf_atomname, rmsf.results.rmsf])
    rmsf_df = pd.DataFrame(data.T, columns=['resname', 'resnum', 'atomname', 'RMSF']).set_index(['resname', 'resnum', 'atomname'])
    rmsf_df.to_csv((rmsf_df_path := output_path.rstrip("/") + '_rmsf.df.csv'))

    # compile all paths to scores into single 'scorefile'
    scores_fn = os.path.join(pose_dir, "mdanalysis_scores.json")
    scores_df = pd.DataFrame.from_dict(
        {
            "description": [description],
            "location": [gro_file],
            "rmsf_df": rmsf_df_path,
            "reference_distances_df": reference_distances_df_path,
            "chi_df": chi_df_path,
            "internal_ref_df": internal_ref_df_path,
            "internal_df": internal_df_path,
            "ca_internal_df": ca_internal_df_path,
            "ca_internal_ref_df": ca_internal_ref_df_path,
            "rmsd_df_path": rmsd_df_path
        }
    )

    # write 'scorefile' for integration to ProtFlow.
    scores_df.to_json(scores_fn)

# %%

if __name__ == "__main__":
    fire.Fire(main)
