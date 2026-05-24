"""
InSilico BioSphere - AI Small-Molecule Redesign Studio
Developed by: Mr. Sarang S. Dhote
Assistant Professor, Department of Chemistry
Shivaji Science College, Nagpur, India
"""

import streamlit as st
import os
import urllib.request
import numpy as np
import pandas as pd
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Geometry import Point3D

# --- LIVE HARDWARE-ACCELERATED 3D RENDER INTERFACE LAYER ---
try:
    import py3Dmol
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

# --- AUTODOCK VINA INTEGRATION CHECK ---
try:
    from vina import Vina
    from meeko import MoleculePreparation
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

# --- OPENBABEL CHECK FOR PROTEIN CONVERSION ---
try:
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False


# --- INITIALIZATION SAFETY WRAPPER ---
def initialize_session():
    defaults = {
        "rd_receptor": None,
        "temp_pdb_path": None,
        "rd_ligand": None,
        "rd_parent_smiles": None,
        "rd_library": None,
        "docking_results": None,
        "protein_parsed": False,
        "ligand_parsed": False,
        "vina_poses_pdbqt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# --- BIOINFORMATICS STRUCTURAL ENGINE ---

def fetch_pdb_from_rcsb(pdb_id):
    """Download PDB file from RCSB Protein Data Bank."""
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4:
        return False, "PDB ID must be exactly 4 characters."
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception as e:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'. {str(e)}"


def prepare_receptor_to_pdbqt(input_pdb_path):
    """Converts a raw PDB to a Vina-ready PDBQT using OpenBabel."""
    if not OPENBABEL_AVAILABLE:
        return False, ("Error: The 'openbabel' python package is required for "
                       "automatic PDB to PDBQT conversion. Please upload a "
                       "pre-prepared .pdbqt file instead.")

    output_pdbqt = input_pdb_path.replace(".pdb", ".pdbqt")
    if input_pdb_path.endswith(".pdbqt"):
        return True, input_pdb_path

    try:
        mols = list(pybel.readfile("pdb", input_pdb_path))
        if not mols:
            return False, "Failed to read PDB file."
        mol = mols[0]

        mol.removeh()
        mol.addh()
        mol.calccharges("gasteiger")
        mol.write("pdbqt", output_pdbqt, overwrite=True)
        return True, output_pdbqt
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"


def auto_detect_heteroatom_center(pdb_path):
    """Find the geometric center of bound ligands/cofactors (excluding water)."""
    coords = []
    if pdb_path and os.path.exists(pdb_path):
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("HETATM") and "HOH" not in line:
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append((x, y, z))
                    except ValueError:
                        continue
    if coords:
        mean_coords = np.mean(coords, axis=0)
        return round(mean_coords[0], 3), round(mean_coords[1], 3), round(mean_coords[2], 3)
    return 0.0, 0.0, 0.0


def detect_contact_residues(receptor_path, pose_pdbqt_string, cutoff=4.0, max_residues=5):
    """Detect actual contact residues within cutoff angstroms of the docked ligand."""
    try:
        ligand_coords = []
        for line in pose_pdbqt_string.split('\n'):
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ligand_coords.append((x, y, z))
                except (ValueError, IndexError):
                    continue
        if not ligand_coords:
            return ["UNK-0"]

        ligand_arr = np.array(ligand_coords)
        residue_min_dist = {}

        with open(receptor_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        resname = line[17:20].strip()
                        resnum = line[22:26].strip()
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atom = np.array([x, y, z])
                        dists = np.linalg.norm(ligand_arr - atom, axis=1)
                        min_d = float(np.min(dists))
                        key = f"{resname}-{resnum}"
                        if key not in residue_min_dist or min_d < residue_min_dist[key]:
                            residue_min_dist[key] = min_d
                    except (ValueError, IndexError):
                        continue

        contacts = [(k, v) for k, v in residue_min_dist.items() if v <= cutoff]
        contacts.sort(key=lambda x: x[1])
        if not contacts:
            return ["No contacts within cutoff"]
        return [c[0] for c in contacts[:max_residues]]
    except Exception:
        return ["UNK-0"]


def run_strict_vina_docking(smiles, receptor_path, cx, cy, cz, box_size=22,
                            exhaustiveness=8, n_poses=5):
    """Run real AutoDock Vina docking."""
    if not VINA_AVAILABLE:
        return False, "Error: Vina or Meeko python packages are not installed."

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES string."
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) < 0:
            return False, "3D embedding failed for ligand."
        AllChem.MMFFOptimizeMolecule(mol)

        prep = MoleculePreparation()
        prep.prepare(mol)
        ligand_pdbqt = prep.write_pdbqt_string()

        v = Vina(sf_name='vina')
        v.set_receptor(receptor_path)
        v.set_ligand_from_string(ligand_pdbqt)

        v.compute_vina_maps(center=[cx, cy, cz], box_size=[box_size, box_size, box_size])
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        energies = v.energies(n_poses=n_poses)
        docked_pdbqt_string = v.poses(n_poses=n_poses)

        contacts = detect_contact_residues(receptor_path, docked_pdbqt_string)

        return True, {
            "energies": energies,
            "poses": docked_pdbqt_string,
            "residues": contacts,
        }
    except Exception as e:
        return False, f"Vina Engine Error: {str(e)}"


def simulate_docking_fallback(smiles, n_poses=5):
    """When Vina isn't available, return a clearly labeled estimate using
    a simple MW/LogP heuristic so the UI remains demonstrable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES."
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    base_score = -5.0 - (mw / 500.0) - abs(logp) * 0.3
    energies = [[round(base_score - (i * 0.4), 2)] for i in range(n_poses)]
    return True, {
        "energies": energies,
        "poses": None,
        "residues": ["DEMO-MODE"] * n_poses,
        "is_simulated": True,
    }


def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    """Render an SMILES string as an inline 2D image."""
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.RemoveHs(mol)
            if include_labels:
                for atom in mol_to_draw.GetAtoms():
                    atom.SetProp('atomNote', str(atom.GetIdx()))
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return (f'<img src="data:image/png;base64,{img_str}" '
                    'style="max-width:100%; border-radius:8px; '
                    'box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>')
    except Exception:
        pass
    return None


def find_valid_cleavage_sites(smiles_str):
    """Identify atoms that are valid targets for substitution."""
    valid_sites = []
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                sym = atom.GetSymbol()
                deg = atom.GetDegree()
                hs = atom.GetTotalNumHs()

                if deg == 1 and sym != 'C':
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} (Terminal {sym})"})
                elif sym == 'C' and hs > 0:
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} ({sym} with available H)"})
                elif sym in ['N', 'O', 'S'] and hs > 0:
                    valid_sites.append({"index": idx, "label": f"Atom #{idx} (Core {sym} with available H)"})

        valid_sites.sort(key=lambda x: (0 if "Terminal" in x["label"] else 1, x["index"]))
    except Exception:
        pass
    return valid_sites


def get_dynamic_fragments(parent_smiles):
    """Classify the parent scaffold and return tailored fragments.

    Note: Yields, FTIR peaks, and synthesis routes are pedagogical heuristics
    intended for teaching scaffold-class trends, not literature-verified data.
    """
    mol = Chem.MolFromSmiles(parent_smiles)
    if not mol:
        return "Standard Organic Scaffold", []

    flavone_smarts = Chem.MolFromSmarts("c1cc(O)cc2c1c(=O)cc(c2)c3ccccc3")
    phenol_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
    alkaloid_smarts = Chem.MolFromSmarts("[#7;R]")
    aliphatic_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.GetIsAromatic()]
    total_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    aliphatic_ratio = len(aliphatic_carbons) / len(total_carbons) if total_carbons else 0

    if mol.HasSubstructMatch(flavone_smarts) or phenol_count >= 2:
        subclass_title = "Polyphenolic Flavonoid Core"
        fragments = [
            {"name": "Glucosylation (-C6H11O5)", "smiles": "OC1C(O)C(O)C(O)C(CO)O1",
             "peak": 3350, "yield": "Moderate Yield (58%)",
             "route": "Enzymatic glycosylation via Phase II transferase mirroring."},
            {"name": "Prenylation (-CH2CH=C(CH3)2)", "smiles": "CC(C)=CC",
             "peak": 1660, "yield": "Good Yield (72%)",
             "route": "Late-stage electrophilic C-alkylation."},
            {"name": "O-Methylation (-OCH3)", "smiles": "OC",
             "peak": 1250, "yield": "Excellent Yield (91%)",
             "route": "Selective etherification using Dimethyl Sulfate."},
            {"name": "Acetylation (-OCOCH3)", "smiles": "OC(=O)C",
             "peak": 1735, "yield": "Good Yield (84%)",
             "route": "Esterification utilizing Acetic Anhydride."},
        ]
    elif mol.HasSubstructMatch(alkaloid_smarts):
        subclass_title = "Alkaloidal Nitrogen Heterocycle"
        fragments = [
            {"name": "N-Alkylation (-CH2CH3)", "smiles": "CC",
             "peak": 2960, "yield": "Good Yield (80%)",
             "route": "Nucleophilic substitution at nitrogen nodes using Ethyl Bromide."},
            {"name": "Quaternization (-CH3+)", "smiles": "C",
             "peak": 2850, "yield": "Excellent Yield (94%)",
             "route": "Methylation using Methyl Iodide."},
            {"name": "Amidation (-COCH3)", "smiles": "C(=O)C",
             "peak": 1665, "yield": "Good Yield (78%)",
             "route": "Amide condensation using Acetyl Chloride."},
            {"name": "N-Oxidation (=O)", "smiles": "[O-]",
             "peak": 950, "yield": "Moderate Yield (65%)",
             "route": "Controlled oxidation via mCPBA."},
        ]
    elif aliphatic_ratio > 0.65:
        subclass_title = "Aliphatic Terpenoid Scaffold"
        fragments = [
            {"name": "Epoxidation (=O)", "smiles": "O",
             "peak": 1250, "yield": "Moderate Yield (60%)",
             "route": "Prilezhaev reaction using mCPBA across isolated alkene bonds."},
            {"name": "Hydroxylation (-OH)", "smiles": "O",
             "peak": 3400, "yield": "Poor Yield (42%)",
             "route": "Allylic C-H functionalization driven by Selenium Dioxide."},
            {"name": "Ozonolysis Fragmentation", "smiles": "O=C",
             "peak": 1710, "yield": "Good Yield (70%)",
             "route": "Oxidative cleavage of double bonds."},
            {"name": "Esterification (-COOCH3)", "smiles": "C(=O)OC",
             "peak": 1740, "yield": "Good Yield (86%)",
             "route": "Fischer esterification across terminal carboxylic vectors."},
        ]
    else:
        subclass_title = "Standard Organic Lead Profile"
        fragments = [
            {"name": "Methylation (-CH3)", "smiles": "C",
             "peak": 2925, "yield": "Good Yield (85%)",
             "route": "Standard alkylation path via Methyl Iodide."},
            {"name": "Hydroxylation (-OH)", "smiles": "O",
             "peak": 3450, "yield": "Moderate Yield (62%)",
             "route": "Direct C-H matrix oxidation with copper coordination."},
            {"name": "Amination (-NH2)", "smiles": "N",
             "peak": 3320, "yield": "Good Yield (74%)",
             "route": "Controlled substitution via nucleophilic amination."},
            {"name": "Fluorination (-F)", "smiles": "F",
             "peak": 1150, "yield": "Poor Yield (38%)",
             "route": "Late-stage electrophilic fluorination using Selectfluor."},
        ]
    return subclass_title, fragments


def run_cleaving_engine(parent_smiles, target_atom_idx, mechanism_mode):
    """Perform structural substitution or non-covalent co-formulation."""
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol:
        return []

    _, fragments = get_dynamic_fragments(parent_smiles)
    derived_library = []

    for idx, frag in enumerate(fragments):
        success = False
        derived_smiles = ""

        if mechanism_mode == "True Covalent Substitution (Cleavage & Attachment)":
            try:
                rw_mol = Chem.RWMol(parent_mol)
                t_atom = rw_mol.GetAtomWithIdx(int(target_atom_idx))
                is_terminal = (t_atom.GetDegree() == 1 and t_atom.GetSymbol() != 'C')

                if is_terminal:
                    t_atom.SetAtomicNum(0)
                    t_atom.SetIsotope(999)
                else:
                    dummy = Chem.Atom(0)
                    dummy.SetIsotope(999)
                    new_idx = rw_mol.AddAtom(dummy)
                    rw_mol.AddBond(int(target_atom_idx), new_idx, Chem.BondType.SINGLE)

                tagged_mol = rw_mol.GetMol()
                Chem.SanitizeMol(tagged_mol)

                pattern = Chem.MolFromSmarts("[999*]")
                frag_mol = Chem.MolFromSmiles(frag['smiles'])

                replaced_mols = AllChem.ReplaceSubstructs(tagged_mol, pattern, frag_mol, replaceAll=True)

                if replaced_mols:
                    final_mol = replaced_mols[0]
                    Chem.SanitizeMol(final_mol)
                    derived_smiles = Chem.MolToSmiles(final_mol)
                    if Chem.MolFromSmiles(derived_smiles):
                        success = True
            except Exception:
                success = False

        if not success:
            derived_smiles = f"{parent_smiles}.{frag['smiles']}"
            frag_name = (frag["name"] + " (Co-Crystal Fallback)"
                         if "Co-Crystal" not in mechanism_mode
                         else frag["name"] + " (Co-Crystal)")
            route = ("Co-crystallization (due to steric constraints blocking covalent bond)."
                     if "Co-Crystal" not in mechanism_mode
                     else "Co-crystallization or therapeutic salt formulation protocol.")
        else:
            frag_name = frag["name"]
            route = frag["route"]

        test_mol = Chem.MolFromSmiles(derived_smiles)
        mw = round(Descriptors.MolWt(test_mol), 2) if test_mol else 0
        logp = round(Descriptors.MolLogP(test_mol), 2) if test_mol else 0

        derived_library.append({
            "Variant ID": f"Derivative-{idx+1:02d}" if success else f"Formulation-{idx+1:02d}",
            "Fragment Added": frag_name,
            "Redesigned SMILES": derived_smiles,
            "MW (g/mol)": mw,
            "LogP": logp,
            "Yield Prediction": frag["yield"] if success else "Pharmaceutical Salt Matrix",
            "Route": route,
            "FTIR Peak": int(frag["peak"]),
        })

    return derived_library


# ============================================================
# APPLICATION SETUP
# ============================================================
st.set_page_config(page_title="InSilico BioSphere Redesign", layout="wide")
st.title("🧬 InSilico BioSphere — AI Small-Molecule Redesign Studio")
st.markdown(
    "**InSilico BioSphere** | Developed by: Mr. Sarang S. Dhote, "
    "Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India"
)

# Environment diagnostics in a collapsible panel
with st.expander("⚙️ Environment Status", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("AutoDock Vina", "✅ Available" if VINA_AVAILABLE else "❌ Missing")
    c2.metric("OpenBabel", "✅ Available" if OPENBABEL_AVAILABLE else "❌ Missing")
    c3.metric("py3Dmol / stmol", "✅ Available" if STMOL_AVAILABLE else "❌ Missing")
    if not VINA_AVAILABLE:
        st.info("Vina is not installed — docking will run in demo (heuristic) mode. "
                "To enable real docking, deploy via Docker (see README).")
    if not OPENBABEL_AVAILABLE:
        st.info("OpenBabel is not installed — please upload pre-prepared .pdbqt receptor files.")

if st.button("🔄 Reset Entire Redesign Environment", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.write("---")

col_params, col_visuals = st.columns([1, 1])

# ============================================================
# LEFT COLUMN — PARAMETER PANEL
# ============================================================
with col_params:
    st.header("1. Target Protein Setup")

    if st.session_state.protein_parsed and st.session_state.rd_receptor:
        st.success(f"🟢 Active Target Matrix: `{os.path.basename(st.session_state.rd_receptor)}`")

    protein_mode = st.radio(
        "Protein Input Setup:",
        ["Download PDB ID", "Upload Local Structure File (.PDB / .PDBQT)"],
    )

    if protein_mode == "Download PDB ID":
        pdb_id = st.text_input("Enter 4-Letter PDB Code", value="2AMB").strip()
        if st.button("📥 Fetch PDB"):
            with st.spinner("Downloading structure from RCSB..."):
                ok, path = fetch_pdb_from_rcsb(pdb_id)
                if ok:
                    st.session_state.temp_pdb_path = path
                    st.session_state.protein_parsed = False
                    st.session_state.rd_receptor = None
                    st.rerun()
                else:
                    st.error(path)
    else:
        uploaded_rec = st.file_uploader("Upload Macromolecule", type=["pdb", "pdbqt"])
        if uploaded_rec:
            path = f"temp_{uploaded_rec.name}"
            if st.session_state.temp_pdb_path != path:
                with open(path, "wb") as f:
                    f.write(uploaded_rec.getbuffer())
                st.session_state.temp_pdb_path = path
                st.session_state.protein_parsed = False
                st.session_state.rd_receptor = None
                st.rerun()

    # Process the uploaded/downloaded file
    if st.session_state.temp_pdb_path and not st.session_state.protein_parsed:
        st.info(f"📂 File Ready for Processing: `{os.path.basename(st.session_state.temp_pdb_path)}`")

        if st.session_state.temp_pdb_path.endswith(".pdbqt"):
            if st.button("✅ Confirm & Load Matrix", type="primary"):
                st.session_state.rd_receptor = st.session_state.temp_pdb_path
                st.session_state.protein_parsed = True
                st.rerun()
        else:
            if OPENBABEL_AVAILABLE:
                if st.button("⚙️ Convert PDB to PDBQT & Load", type="primary"):
                    with st.spinner("Converting structure via OpenBabel..."):
                        conv_ok, final_path = prepare_receptor_to_pdbqt(st.session_state.temp_pdb_path)
                        if conv_ok:
                            st.session_state.rd_receptor = final_path
                            st.session_state.protein_parsed = True
                            st.session_state.temp_pdb_path = None
                            st.rerun()
                        else:
                            st.error(final_path)
            else:
                st.warning("⚠️ OpenBabel not available. You can still load the PDB for visualization, "
                           "but real Vina docking requires a .pdbqt file.")
                if st.button("📋 Load PDB for Visualization Only"):
                    st.session_state.rd_receptor = st.session_state.temp_pdb_path
                    st.session_state.protein_parsed = True
                    st.rerun()

    st.write("---")
    st.header("2. Phytochemical Scaffold Profile")

    if st.session_state.ligand_parsed and st.session_state.rd_parent_smiles:
        st.success("🟢 Phytochemical Lead Scaffold Coordinates Ready")

    smiles_input = st.text_input(
        "Parent Compound SMILES",
        value="CC(=O)NC1=CC=C(O)C=C1",
        help="Default is Paracetamol. Try quercetin: O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    ).strip()

    if st.button("📥 Load Phytochemical Scaffold Profile"):
        test_mol = Chem.MolFromSmiles(smiles_input)
        if test_mol is None:
            st.error("Invalid SMILES string. Please check the syntax.")
        else:
            st.session_state.rd_parent_smiles = smiles_input
            st.session_state.ligand_parsed = True
            st.rerun()

    if st.session_state.protein_parsed and st.session_state.ligand_parsed and st.session_state.rd_parent_smiles:
        st.write("---")
        st.header("3. Reaction Mechanism")

        valid_sites = find_valid_cleavage_sites(st.session_state.rd_parent_smiles)
        subclass, _ = get_dynamic_fragments(st.session_state.rd_parent_smiles)
        st.caption(f"🧪 Detected scaffold class: **{subclass}**")

        if len(valid_sites) == 0:
            st.warning("⚠️ High Steric Hindrance: Enforcing Co-Crystal mode.")
            reaction_mode = "Co-Crystal / Salt Formulation (Non-Covalent)"
        else:
            reaction_mode = st.radio(
                "Select Modification Mechanism:",
                ["True Covalent Substitution (Cleavage & Attachment)",
                 "Co-Crystal / Salt Formulation (Non-Covalent)"],
            )

        show_labels = st.toggle("🔍 Show Atom Index Numbers on Structure", value=True)
        base_img = generate_clean_2d_image(
            st.session_state.rd_parent_smiles,
            include_labels=show_labels,
            zoom_level=600,
        )
        if base_img:
            st.html(base_img)

        if reaction_mode == "True Covalent Substitution (Cleavage & Attachment)" and valid_sites:
            st.info("💡 Select an atom from the list below.")
            site_options = {site["label"]: site["index"] for site in valid_sites}
            selected_site_label = st.selectbox(
                "🎯 Select Valid Target Atom for Substitution",
                options=list(site_options.keys()),
            )
            target_idx = site_options[selected_site_label]
        else:
            target_idx = 0

        if st.button("🚀 Generate Structural Derivatives"):
            st.session_state.docking_results = None
            st.session_state.vina_poses_pdbqt = None
            with st.spinner("Processing structural operations..."):
                results_list = run_cleaving_engine(
                    st.session_state.rd_parent_smiles, target_idx, reaction_mode
                )
                if len(results_list) > 0:
                    st.session_state.rd_library = pd.DataFrame(results_list)
                    st.rerun()
                else:
                    st.error("Structural substitution failed.")


# ============================================================
# RIGHT COLUMN — SCREENING & DOCKING WORKSPACE
# ============================================================
with col_visuals:
    st.header("4. Screening & Docking Workspace")

    if (st.session_state.protein_parsed
        and st.session_state.ligand_parsed
        and st.session_state.rd_library is not None):

        st.dataframe(
            st.session_state.rd_library[["Variant ID", "Fragment Added", "MW (g/mol)", "LogP"]],
            hide_index=True,
            use_container_width=True,
        )

        # Full derivative download
        csv = st.session_state.rd_library.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Full Derivative Library (CSV)",
            csv,
            "derivatives.csv",
            "text/csv",
        )

        st.write("---")
        st.subheader("🔍 Selection Isolation & 2D Topography")
        chosen_variant_id = st.selectbox(
            "Select variant for docking:",
            options=st.session_state.rd_library["Variant ID"],
        )

        selected_rows = st.session_state.rd_library[
            st.session_state.rd_library["Variant ID"] == chosen_variant_id
        ]

        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]

            with st.expander("📋 Derivative details", expanded=False):
                st.write(f"**SMILES:** `{selected_row['Redesigned SMILES']}`")
                st.write(f"**Route:** {selected_row['Route']}")
                st.write(f"**Yield Prediction:** {selected_row['Yield Prediction']}")
                st.write(f"**FTIR Peak:** {selected_row['FTIR Peak']} cm⁻¹")

            highlighted_img_html = generate_clean_2d_image(str(selected_row["Redesigned SMILES"]))
            if highlighted_img_html:
                st.html(highlighted_img_html)

            st.write("---")
            st.header("🚀 5. Docking Engine")

            det_x, det_y, det_z = auto_detect_heteroatom_center(st.session_state.rd_receptor)
            st.info(f"Auto-detected active site center → X: {det_x}, Y: {det_y}, Z: {det_z}")

            with st.expander("⚙️ Advanced docking parameters", expanded=False):
                cx = st.number_input("Box center X", value=float(det_x))
                cy = st.number_input("Box center Y", value=float(det_y))
                cz = st.number_input("Box center Z", value=float(det_z))
                box_size = st.slider("Search box size (Å)", 10, 40, 22)
                exhaustiveness = st.slider("Exhaustiveness", 1, 32, 8)
                n_poses = st.slider("Number of poses", 1, 20, 5)

            if VINA_AVAILABLE:
                button_label = "🚀 Run TRUE Vina Docking (1–3 minutes)"
            else:
                button_label = "🧪 Run Heuristic Demo Docking (Vina unavailable)"

            if st.button(button_label):
                if VINA_AVAILABLE:
                    with st.spinner("Running AutoDock Vina..."):
                        success, v_data = run_strict_vina_docking(
                            str(selected_row["Redesigned SMILES"]),
                            st.session_state.rd_receptor,
                            cx, cy, cz, box_size, exhaustiveness, n_poses,
                        )
                else:
                    with st.spinner("Computing heuristic estimate..."):
                        success, v_data = simulate_docking_fallback(
                            str(selected_row["Redesigned SMILES"]), n_poses
                        )

                if success:
                    st.session_state.vina_poses_pdbqt = v_data.get("poses")
                    pose_list = []
                    for p in range(len(v_data["energies"])):
                        residues = v_data["residues"]
                        pose_list.append({
                            "Pose ID": f"Pose #{p+1}",
                            "Energy": round(v_data["energies"][p][0], 2),
                            "Pose Rank": p,
                            "Residue": residues[p % len(residues)],
                        })
                    st.session_state.docking_results = pose_list
                    if v_data.get("is_simulated"):
                        st.warning("⚠️ Demo mode: these scores are heuristic estimates, not real docking results.")
                    else:
                        st.success("✅ Vina Docking Complete!")
                else:
                    st.error(v_data)

            if st.session_state.docking_results is not None:
                st.write("---")
                st.subheader("📊 Pose Analysis")

                results_df = pd.DataFrame(st.session_state.docking_results)
                st.dataframe(
                    results_df[["Pose ID", "Energy", "Residue"]],
                    hide_index=True,
                    use_container_width=True,
                )

                pose_options = [p["Pose ID"] for p in st.session_state.docking_results]
                selected_pose_name = st.selectbox(
                    "🎯 Select Docking Pose to Inspect",
                    options=pose_options,
                )
                selected_pose_data = next(
                    item for item in st.session_state.docking_results
                    if item["Pose ID"] == selected_pose_name
                )

                col_a, col_b = st.columns(2)
                col_a.metric("Binding Affinity", f"{selected_pose_data['Energy']} kcal/mol")
                col_b.metric("Nearest Residue", selected_pose_data['Residue'])

                # 3D visualization
                if (STMOL_AVAILABLE
                    and st.session_state.rd_receptor
                    and st.session_state.vina_poses_pdbqt):

                    st.write("---")
                    st.subheader(f"🖥️ 3D Interaction View — {selected_pose_name}")

                    xyz_view = py3Dmol.view(width=700, height=500)

                    with open(st.session_state.rd_receptor, "r") as pf:
                        format_str = "pdbqt" if st.session_state.rd_receptor.endswith(".pdbqt") else "pdb"
                        xyz_view.addModel(pf.read(), format_str)

                    xyz_view.setStyle({'model': 0}, {'cartoon': {'color': 'white', 'opacity': 0.4}})
                    xyz_view.addSurface(py3Dmol.VDW, {'opacity': 0.1, 'color': 'white'}, {'model': 0})

                    res_info = selected_pose_data.get('Residue', 'UNK-0')
                    try:
                        res_num = int(res_info.split('-')[1])
                    except (IndexError, ValueError):
                        res_num = -1
                    if res_num != -1:
                        xyz_view.addStyle(
                            {'model': 0, 'resi': str(res_num)},
                            {'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15}},
                        )
                        xyz_view.addLabel(
                            f"Contact: {res_info}",
                            {'fontColor': 'orange', 'backgroundColor': 'white',
                             'showBackground': True, 'fontSize': 12},
                            {'model': 0, 'resi': str(res_num)},
                        )

                    xyz_view.addModelsAsFrames(st.session_state.vina_poses_pdbqt, "pdbqt")

                    frame_idx = selected_pose_data['Pose Rank']
                    xyz_view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
                    xyz_view.addStyle({'model': 1}, {'sphere': {'radius': 0.35, 'colorscheme': 'greenCarbon'}})

                    xyz_view.setFrame(frame_idx, {'model': 1})
                    xyz_view.zoomTo({'model': 1})

                    showmol(xyz_view, height=500, width=700)
                elif not STMOL_AVAILABLE:
                    st.info("Install py3Dmol and stmol to enable 3D visualization.")
                elif not st.session_state.vina_poses_pdbqt:
                    st.info("3D pose viewer requires real Vina output. Currently running in demo mode.")

    else:
        st.info("📊 Workspace Gated: Please load both protein and ligand to proceed.")
        if not st.session_state.protein_parsed:
            st.write("- ⏳ Waiting for protein...")
        if not st.session_state.ligand_parsed:
            st.write("- ⏳ Waiting for ligand SMILES...")
        if st.session_state.rd_library is None and st.session_state.ligand_parsed:
            st.write("- ⏳ Waiting for derivative generation...")


# ============================================================
# FOOTER
# ============================================================
st.write("---")
st.caption(
    "⚠️ **Disclaimer:** Synthesis routes, yield predictions, and FTIR peaks shown for "
    "fragments are pedagogical heuristics intended to illustrate scaffold-class trends — "
    "they are not literature-validated values. Docking results from AutoDock Vina are real "
    "when Vina is available; otherwise the app runs in clearly-labeled demo mode."
)
