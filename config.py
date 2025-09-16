import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings


class Residue(BaseModel):
    chain: str
    index: int

    def __str__(self):
        return f"{self.chain}{self.index}"

    @classmethod
    def parse(cls, value: str) -> Self:
        """Extract chain identifier and residue number from a PDB numbering string.

        Args:
            value (str): PDB numbering string, e.g., "A123" or "123".

        Returns:
            Residue object

        Raises:
            ValueError: If the value format is invalid.

        Examples:
            >>> Residue.parse("A123")
            Residue(chain='A', index=123)
            >>> Residue.parse("123")
            Residue(chain='A', index=123)
            >>> Residue.parse("AB12")
            Residue(chain='AB', index=12)
            >>> Residue.parse("A12B")
            Traceback (most recent call last):
            ValueError: Invalid pdbnum format: A12B
        """
        match = re.match(r"^(?P<chain>[A-Za-z]*)(?P<resnum>\d+)$", value)
        if not match:
            raise ValueError(f"Invalid pdbnum format: {value}")
        return cls(chain=match.group("chain") or "A", index=int(match.group("resnum")))


class CovalentBond(BaseModel):
    residue_atom: str
    ligand: Residue
    ligand_atom: str

    def __str__(self):
        return f"{self.residue_atom}:{self.ligand}-{self.ligand_atom}"

    @classmethod
    def parse(cls, value: str) -> Self:
        residue_atom, ligand_info = value.split(":", maxsplit=1)
        ligand, ligand_atom = ligand_info.split("-", maxsplit=1)
        return cls(
            residue_atom=residue_atom,
            ligand=Residue.parse(ligand),
            ligand_atom=ligand_atom,
        )


class CommonArguments(BaseSettings, use_attribute_docstrings=True):
    riff_diff_dir: Path = Field(default_factory=Path.cwd)
    """Path to the riff_diff directory."""

    output_prefix: str | None = None
    """Prefix for all output files."""

    fragment_pdb: Path | None = None
    """
    Path to backbone fragment pdb. If not set, an idealized 7-residue helix fragment is
    used.
    """

    pick_frags_from_db: bool = False
    """
    Select backbone fragments from database instead of providing a specific backbone
    manually.
    
    WARNING: This is much more time consuming!
    """

    custom_channel_path: Path | None = None
    """Use a custom channel placeholder. Must be the path to a .pdb file."""

    channel_chain: str | None = None
    """
    Chain of the custom channel placeholder (if using a custom channel specified with
    <custom_channel_path>)
    """

    preserve_channel_coordinates: bool = False
    """
    Copies channel from channel reference pdb without superimposing on moitf-substrate
    centroid axis. Useful when channel is present in catalytic array.
    """

    rotamer_positions: Literal["auto"] | list[int] | int = "auto"
    """
    Position in fragment the rotamer should be inserted, can either be int or a list
    containing first and last position (e.g. 2,6 if rotamer should be inserted at every
    position from 2 to 6). Recommended not to include N- and C-terminus! If auto,
    rotamer is inserted at every position (except N- and C-terminus).
    """

    rmsd_cutoff: float = 0.5
    """
    Set minimum RMSD of output fragments. Increase to get more diverse fragments, but
    high values might lead to very long runtime or few fragments!
    """

    prob_cutoff: float = 0.05
    """
    Do not return any phi/psi combinations with chi angle probabilities below this
    value
    """

    max_frags_per_residue: int = 150
    """Maximum number of fragments that should be returned per active site residue."""

    add_equivalent_func_groups: bool = False
    """use ASP/GLU, GLN/ASN and VAL/ILE interchangeably."""

    rot_lig_clash_vdw_multiplier: float = 0.8
    """
    Multiplier for Van-der-Waals radii for clash detection between rotamer and ligand.
    Functional groups are not checked! Clash is detected if a distance between
    atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.
    """

    bb_lig_clash_vdw_multiplier: float = 1.0
    """
    Multiplier for Van-der-Waals radii for clash detection between fragment backbone
    and ligand. Clash is detected if a distance between
    atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.
    """

    channel_frag_clash_vdw_multiplier: float = 1.0
    """
    Multiplier for Van-der-Waals radii for clash detection between fragment backbone
    and channel placeholder. Clash is detected if a distance between
    atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.
    """

    fragsize: int = 7
    """Size of output fragments. Only used if <pick_frags_from_db> is set."""

    rot_sec_struct: str | None = None
    """
    Limit fragments to secondary structure at rotamer position. Provide string of
    one-letter code of dssp secondary structure elements (B, E, G, H, I, T, S, -), e.g.
    'HE' if rotamer should be in helices or beta strands. Only used if
    <pick_frags_from_db> is set.
    """

    frag_sec_struct_fraction: str | None = None
    """
    Limit to fragments containing at least fraction of residues with the provided
    secondary structure. If fragment should have at least 50 percent helical residues
    OR 60 percent beta-sheet, pass 'H:0.5,E:0.6'. Only used if <pick_frags_from_db> is
    set.
    """

    phipsi_occurrence_cutoff: float = 0.5
    """
    Limit how common the phi/psi combination of a certain rotamer has to be. Value is
    in percent.
    """

    jobstarter: Literal["SbatchArray", "Local"] = "SbatchArray"
    """
    Defines if jobs run locally or distributed on a cluster using a protflow
    jobstarter.
    """

    cpus: int = 60
    """Defines how many cpus should be used for distributed computing."""

    rotamer_chi_binsize: float = 10.0
    """
    Filter for diversifying found rotamers. Lower numbers mean more similar rotamers
    will be found. Similar rotamers will still be accepted if their backbone angles are
    different (if rotamer_phipsi_bin is set).
    """

    rotamer_phipsi_binsize: float = 20.0
    """
    Filter for diversifying found rotamers. Lower numbers mean similar rotamers from
    more similar backbone angles will be accepted.
    """

    phi_psi_bin: Annotated[float, Field(le=10.0)] = 9.9
    """
    Binsize used to identify if fragment fits to phi/psi combination. Should not be
    above 10!
    """

    max_rotamers: int = 80
    """
    maximum number of phi/psi combination that should be returned. Can be increased if
    not enough fragments are found downstream (e.g. because secondary structure filter
    was used, and there are not enough phi/psi combinations in the output that fit to
    the specified secondary structure.
    """

    rotamer_diff_to_best: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    """
    Accept rotamers that have a probability not lower than this percentage of the most
    probable accepted rotamer. 1 means all rotamers will be accepted.
    """

    not_flip_symmetric: bool = False
    """Do not flip tip symmetric residues (ARG, ASP, GLU, LEU, PHE, TYR, VAL)."""

    prob_weight: float = 2.0
    """Weight for rotamer probability importance when picking rotamers."""

    occurrence_weight: float = 1.0
    """Weight for phi/psi-occurrence importance when picking rotamers."""

    backbone_score_weight: float = 1.0
    """
    Weight for importance of fragment backbone score (boltzman score of number of
    occurrences of similar fragments in the database) when sorting fragments.
    """

    rotamer_score_weight: float = 1.0
    """
    Weight for importance of rotamer score (combined score of probability and
    occurrence) when sorting fragments.
    """

    chi_std_multiplier: float = 2.0
    """
    Multiplier for chi angle standard deviation to check if rotamer in database fits to
    desired rotamer.
    """

    max_paths_per_ensemble: int = 4
    """Maximum number of paths per ensemble (=same fragments but in different order)"""

    frag_frag_bb_clash_vdw_multiplier: float = 0.9
    """
    Multiplier for VanderWaals radii for clash detection inbetween backbone fragments.
    Clash is detected if
    distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.
    """

    frag_frag_sc_clash_vdw_multiplier: float = 0.8
    """
    Multiplier for VanderWaals radii for clash detection between fragment sidechains
    and backbones. Clash is detected if distance_between_atoms < (VdW_radius_atom1 +
    VdW_radius_atom2)*multiplier.
    """

    fragment_score_weight: float = 1.0
    """Maximum number of cpus to run on"""

    max_top_out: int = 100
    """Maximum number of top-ranked output paths"""

    max_random_out: int = 100
    """Maximum number of random-ranked output paths"""

    def model_post_init(self, context):
        self.riff_diff_dir = self.riff_diff_dir.resolve()


class ResArguments(CommonArguments, use_attribute_docstrings=True):
    theozyme_pdb: Path | None = None
    """Path to pdbfile containing theozyme."""

    working_dir: Path | None = None
    """Output directory"""

    covalent_bonds: list[CovalentBond] | None = None
    """
    Add covalent bond(s) between residues and ligands in the form
    'Res1-Res1Atom:Lig1-Lig1Atom,Res2-Res2Atom:Lig2-Lig2Atom'.
    
    Atom names should follow PDB numbering schemes.
    
    Example:
    'A23-NE2:Z1-C1 A26-OE1:Z1-C11' for two covalent bonds between the NE2 atom of a
    Histidine at position A23 to C1 atom of ligand Z1 and the OE1 atom of a glutamic
    acid at A26 to C11 on the same ligand.
    """

    @field_validator("covalent_bonds", mode="before")
    @classmethod
    def validate_covalent_bonds(cls, v: Any):
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            v = [CovalentBond.parse(i) if isinstance(i, str) else i for i in v]
        return v


class Arguments(
    CommonArguments,
    extra="ignore",
    use_attribute_docstrings=True,
):
    theozyme_pdb: Path
    """Path to pdbfile containing theozyme."""

    working_dir: Path
    """Output directory"""

    theozyme_resnums: list[Residue]
    """
    List of residue numbers with chain information (e.g. 'A25 A38 B188') in theozyme
    pdb to find fragments for.
    """

    ligands: list[Residue]
    "List of ligands in theozyme pdb with chain information in the format 'X188 Z1'."

    res_args: dict[str, ResArguments] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def validate_arguments(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Any:
        model = handler(data)
        if not isinstance(data, dict):
            return model
        theozyme_resnums = [str(r) for r in model.theozyme_resnums]
        for key, value in data.items():
            if key in theozyme_resnums:
                model.res_args[key] = ResArguments.model_validate(value)
        return model

    @field_validator("ligands", "theozyme_resnums", mode="before")
    @classmethod
    def validate_ligands(cls, v: Any):
        if isinstance(v, list):
            v = [Residue.parse(i) if isinstance(i, str) else i for i in v]
        return v

    @field_serializer("ligands", "theozyme_resnums", when_used="json")
    def serialize_ligands(self, ligands: list[tuple[str, int]]):
        return [str(ligand) for ligand in ligands]

    def get_res_args(self, resname: str):
        if resname not in self.res_args:
            return ResArguments.model_validate(
                self.model_dump(
                    exclude={"theozyme_resnums", "ligands", "res_args"},
                    exclude_unset=True,
                ),
            )
        return ResArguments.model_validate(
            self.model_dump(
                exclude={"theozyme_resnums", "ligands", "res_args"}, exclude_unset=True
            )
            | self.res_args[resname].model_dump()
        )
