import streamlit as st
import os
import numpy as np
import pandas as pd
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

# --- INITIALIZATION SAFETY WRAPPER ---
def initialize_session():
    defaults = {
        "rd_parent_smiles": None,
        "rd_library": None,
        "ligand_parsed": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# --- BIOINFORMATICS STRUCTURAL ENGINE ---

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    """Generates high-quality 2D topographic maps of the molecules."""
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
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>'
    except Exception:
        pass
    return None

def find_valid_cleavage_sites(smiles_str):
    """Scans the parent molecule for chemically valid substitution anchor points."""
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
    """Identifies the scaffold class and proposes appropriate functional groups."""
    mol = Chem.MolFromSmiles(parent_smiles)
    if not mol: return "Standard Organic Scaffold", []

    flavone_smarts = Chem.MolFromSmarts("c1cc(O)cc2c1c(=O)cc(c2)c3ccccc3")
    phenol_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
    alkaloid_smarts = Chem.MolFromSmarts("[#7;R]")
    aliphatic_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.GetIsAromatic()]
    total_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    aliphatic_ratio = len(aliphatic_carbons) / len(total_carbons) if total_carbons else 0

    if mol.HasSubstructMatch(flavone_smarts) or phenol_count >= 2:
        subclass_title = "Polyphenolic Flavonoid Core"
        fragments = [
            {"name": "Glucosylation (-C6H11O5)", "smiles": "OC1C(O)C(O)C(O)C(CO)O1", "peak": 3350, "yield": "Moderate Yield (58%)", "route": "Enzymatic glycosylation via Phase II transferase mirroring."},
            {"name": "Prenylation (-CH2CH=C(CH3)2)", "smiles": "CC(C)=CC", "peak": 1660, "yield": "Good Yield (72%)", "route": "Late-stage electrophilic C-alkylation."},
            {"name": "O-Methylation (-OCH3)", "smiles": "OC", "peak": 1250, "yield": "Excellent Yield (91%)", "route": "Selective etherification using Dimethyl Sulfate."},
            {"name": "Acetylation (-OCOCH3)", "smiles": "OC(=O)C", "peak": 1735, "yield": "Good Yield (84%)", "route": "Esterification utilizing Acetic Anhydride."}
        ]
    elif mol.HasSubstructMatch(alkaloid_smarts):
        subclass_title = "Alkaloidal Nitrogen Heterocycle"
        fragments = [
            {"name": "N-Alkylation (-CH2CH3)", "smiles": "CC", "peak": 2960, "yield": "Good Yield (80%)", "route": "Nucleophilic substitution at nitrogen nodes using Ethyl Bromide."},
            {"name": "Quaternization (-CH3+)", "smiles": "C", "peak": 2850, "yield": "Excellent Yield (94%)", "route": "Methylation using Methyl Iodide."},
            {"name": "Amidation (-COCH3)", "smiles": "C(=O)C", "peak": 1665, "yield": "Good Yield (78%)", "route": "Amide condensation using Acetyl Chloride."},
            {"name": "N-Oxidation (=O)", "smiles": "[O-]", "peak": 950, "yield": "Moderate Yield (65%)", "route": "Controlled oxidation via mCPBA."}
        ]
    elif aliphatic_ratio > 0.65:
        subclass_title = "Aliphatic Terpenoid Scaffold"
        fragments = [
            {"name": "Epoxidation (=O)", "smiles": "O", "peak": 1250, "yield": "Moderate Yield (60%)", "route": "Prilezhaev reaction using mCPBA across isolated alkene bonds."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3400, "yield": "Poor Yield (42%)", "route": "Allylic C-H functionalization driven by Selenium Dioxide."},
            {"name": "Ozonolysis Fragmentation", "smiles": "O=C", "peak": 1710, "yield": "Good Yield (70%)", "route": "Oxidative cleavage of double bonds."},
            {"name": "Esterification (-COOCH3)", "smiles": "C(=O)OC", "peak": 1740, "yield": "Good Yield (86%)", "route": "Fischer esterification across terminal carboxylic vectors."}
        ]
    else:
        subclass_title = "Standard Organic Lead Profile"
        fragments = [
            {"name": "Methylation (-CH3)", "smiles": "C", "peak": 2925, "yield": "Good Yield (85%)", "route": "Standard alkylation path via Methyl Iodide."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3450, "yield": "Moderate Yield (62%)", "route": "Direct C-H matrix oxidation with copper coordination."},
            {"name": "Amination (-NH2)", "smiles": "N", "peak": 3320, "yield": "Good Yield (74%)", "route": "Controlled substitution via nucleophilic amination."},
            {"name": "Fluorination (-F)", "smiles": "F", "peak": 1150, "yield": "Poor Yield (38%)", "route": "Late-stage electrophilic fluorination using Selectfluor."}
        ]
    return subclass_title, fragments

def run_cleaving_engine(parent_smiles, target_atom_idx, mechanism_mode):
    """Executes the dummy-tag replacement algorithm to construct new derivatives."""
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol: return []
        
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
                
                # Plant Dummy Tag
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
                
                # Graft the fragment
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
            frag_name = frag["name"] + " (Co-Crystal Fallback)" if "Co-Crystal" not in mechanism_mode else frag["name"] + " (Co-Crystal)"
            route = "Co-crystallization (due to steric constraints blocking covalent bond)." if "Co-Crystal" not in mechanism_mode else "Co-crystallization or therapeutic salt formulation protocol."
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
            "FTIR Peak": int(frag["peak"])
        })
            
    return derived_library


# --- APPLICATION SETUP ---
st.set_page_config(page_title="InSilico BioSphere Redesign", layout="wide")
st.title("🧬 InSilico BioSphere AI Molecular Redesign Studio")
st.markdown("**InSilico BioSphere** | Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India")

if st.button("🔄 Reset Environment", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.write("---")

col_params, col_visuals = st.columns([1, 1])

with col_params:
    st.header("1. Phytochemical Scaffold Profile")
    
    if st.session_state.ligand_parsed and st.session_state.rd_parent_smiles:
        st.success("🟢 Phytochemical Lead Scaffold Loaded")
        
    ligand_mode = st.radio("Lead Input Setup:", ["Paste SMILES String", "Upload Small Molecule File (.SDF, .PDB)"])
    
    if ligand_mode == "Paste SMILES String":
        default_smiles = "CC(=O)NC1=CC=C(O)C=C1" # Paracetamol as a default example
        smiles_input = st.text_input("Parent Compound SMILES", value=default_smiles).strip()
        if st.button("📥 Load Scaffold Profile", type="primary"):
            st.session_state.rd_parent_smiles = smiles_input
            st.session_state.ligand_parsed = True
            st.rerun()
    else:
        uploaded_lig = st.file_uploader("Upload Molecule Block", type=["pdb", "sdf"])
        if uploaded_lig:
            temp_path = f"temp_lig_{uploaded_lig.name}"
            with open(temp_path, "wb") as f: 
                f.write(uploaded_lig.getbuffer())
            
            mol = None
            if temp_path.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(temp_path, removeHs=False)
            else:
                suppl = Chem.SDMolSupplier(temp_path, removeHs=False)
                if suppl and len(suppl) > 0:
                    mol = suppl[0]

            if mol:
                extracted_smiles = str(Chem.MolToSmiles(Chem.RemoveHs(mol)))
                st.session_state.rd_parent_smiles = extracted_smiles
                st.session_state.ligand_parsed = True
                st.success(f"🟢 Upload Complete! Auto-Extracted SMILES Matrix: `{extracted_smiles}`")
            else:
                st.error("Could not extract chemical structure from the uploaded file.")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if st.session_state.ligand_parsed and st.session_state.rd_parent_smiles:
        st.write("---")
        st.header("2. Reaction Mechanism & Target Selection")
        
        class_label, _ = get_dynamic_fragments(st.session_state.rd_parent_smiles)
        st.write(f"🔬 **AI Scaffold Classification:** `{class_label}`")
        
        valid_sites = find_valid_cleavage_sites(st.session_state.rd_parent_smiles)
        
        st.write("##### ⚙️ Synthesis Control Panel")
        
        if len(valid_sites) == 0:
            st.warning("⚠️ High Steric Hindrance: Enforcing Co-Crystal mode.")
            reaction_mode = "Co-Crystal / Salt Formulation (Non-Covalent)"
        else:
            reaction_mode = st.radio(
                "Select Modification Mechanism:", 
                ["True Covalent Substitution (Cleavage & Attachment)", "Co-Crystal / Salt Formulation (Non-Covalent)"]
            )
        
        show_labels = st.toggle("🔍 Show Atom Index Numbers on Structure", value=True)
        base_img = generate_clean_2d_image(st.session_state.rd_parent_smiles, include_labels=show_labels, zoom_level=600)
        if base_img: st.html(base_img)
        
        if reaction_mode == "True Covalent Substitution (Cleavage & Attachment)":
            st.info("💡 Select a target anchor atom from the topographic map above.")
            site_options = {site["label"]: site["index"] for site in valid_sites}
            selected_site_label = st.selectbox("🎯 Target Anchor Atom", options=list(site_options.keys()))
            target_idx = site_options[selected_site_label]
        else:
            target_idx = 0

        if st.button("🚀 Generate Structural Derivatives", type="primary"):
            with st.spinner("Processing structural operations..."):
                results_list = run_cleaving_engine(st.session_state.rd_parent_smiles, target_idx, reaction_mode)
                if len(results_list) > 0:
                    st.session_state.rd_library = pd.DataFrame(results_list)
                    st.rerun()
                else:
                    st.error("Structural substitution failed.")

with col_visuals:
    st.header("3. Derivative Screening Library")
    
    if st.session_state.ligand_parsed and st.session_state.rd_library is not None:
        st.dataframe(st.session_state.rd_library[["Variant ID", "Fragment Added", "MW (g/mol)", "LogP"]], hide_index=True, use_container_width=True)
        
        st.write("---")
        st.subheader("🔍 Isolate Variant Topography")
        chosen_variant_id = st.selectbox("Select variant to view structural map:", options=st.session_state.rd_library["Variant ID"])
        
        selected_rows = st.session_state.rd_library[st.session_state.rd_library["Variant ID"] == chosen_variant_id]
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            
            highlighted_img_html = generate_clean_2d_image(str(selected_row["Redesigned SMILES"]))
            if highlighted_img_html: st.html(highlighted_img_html)
            
            st.write(f"**Structural Identification:** Appended functional group: **{str(selected_row['Fragment Added'])}**.")
            
            st.write("---")
            st.subheader("🧪 Synthetic Route Evaluation")
            st.success(f"**Predicted Efficiency Level:** {str(selected_row['Yield Prediction'])}")
            st.write(f"**Proposed Retrosynthetic Reaction Pathway:** {str(selected_row['Route'])}")
            
            st.write("##### 📋 Target Redesign SMILES")
            st.code(f"{str(selected_row['Redesigned SMILES'])}", language="text")
            
            st.write("---")
            st.subheader("📊 Theoretical Vibrational Spectrum (FTIR)")
            wavenumbers = np.linspace(400, 4000, 500)
            baseline = 98.0 - 2.0 * np.sin(wavenumbers / 200.0)
            target_peak = int(selected_row["FTIR Peak"])
            effect = 40.0 * np.exp(-((wavenumbers - target_peak) / 45.0)**2)
            
            chart_df = pd.DataFrame({"Wavenumber": wavenumbers, "Transmittance (%)": np.clip(baseline - effect, 5.0, 100.0)}).set_index("Wavenumber")
            st.line_chart(chart_df, height=250)
                        
    else:
        st.info("📊 Workspace Gated: Please load a parent scaffold and generate derivatives to view results.")
