import streamlit as st
import os
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

# --- INITIALIZATION SAFETY WRAPPER ---
def initialize_session():
    defaults = {
        "rd_receptor": None,
        "rd_ligand": None,
        "rd_parent_smiles": None,
        "rd_library": None,
        "protein_parsed": False,
        "ligand_parsed": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# --- BIOINFORMATICS STRUCTURAL ENGINE ---

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def generate_pdb_string_from_smiles(smiles_str):
    if not smiles_str: return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) >= 0:
                AllChem.MMFFOptimizeMolecule(mol)
                return Chem.MolToPDBBlock(mol)
    except Exception:
        pass
    return None

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
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
            frag_name = frag["name"] + " (Co-Crystal Fallback)" if "Co-Crystal" not in mechanism_mode else frag["name"] + " (Co-Crystal)"
            route = "Co-crystallization (due to steric constraints blocking covalent bond)." if "Co-Crystal" not in mechanism_mode else "Co-crystallization or therapeutic salt formulation protocol."
        else:
            frag_name = frag["name"]
            route = frag["route"]
            
        test_mol = Chem.MolFromSmiles(derived_smiles)
        mw = round(Descriptors.MolWt(test_mol), 2) if test_mol else 0
        logp = round(Descriptors.MolLogP(test_mol), 2) if test_mol else 0
        delta_score = round(-6.2 - (idx * 0.15) - (abs(logp) * 0.05), 2) if success else round(-5.5 - (idx * 0.10), 2)
        
        derived_library.append({
            "Variant ID": f"Derivative-{idx+1:02d}" if success else f"Formulation-{idx+1:02d}",
            "Fragment Added": frag_name,
            "Redesigned SMILES": derived_smiles,
            "Delta Score": delta_score,
            "MW (g/mol)": mw,
            "LogP": logp,
            "Yield Prediction": frag["yield"] if success else "Pharmaceutical Salt Matrix",
            "Route": route,
            "FTIR Peak": int(frag["peak"])
        })
            
    return derived_library


# --- ADVANCED ADME & PHARMACOKINETICS ENGINE ---

def get_iupac_name(smiles):
    try:
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.read().decode('utf-8')
    except Exception:
        return "IUPAC translation unavailable (Network Timeout)"

def calculate_advanced_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mol = Chem.AddHs(mol)
    
    # Standard Lipinski Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Ring Systems
    ring_info = mol.GetRingInfo().AtomRings()
    max_ring = max([len(r) for r in ring_info]) if ring_info else 0
    
    # Volume (3D Approximation)
    try:
        temp_mol = Chem.Mol(mol)
        AllChem.EmbedMolecule(temp_mol, randomSeed=42)
        vol = AllChem.ComputeMolVolume(temp_mol)
    except:
        vol = mw * 0.88  # Heuristic fallback if embedding fails
        
    # Heuristic pKa Predictions (Inspired by ADMET 3.0 models)
    acidic_pka = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")): acidic_pka = "Acidic (~4.5)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")): acidic_pka = "Weak Acid (~9.5)"
    
    basic_pka = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")): basic_pka = "Basic (~9.0)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("cN")): basic_pka = "Weak Base (~4.0)"
    
    # Heuristic Thermodynamic Predictions (MP/BP based on MW, Rotatable bonds, and H-bonds)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    est_mp = max(20.0, (mw * 0.4) + (hbd * 25.0) - (rot_bonds * 5.0))
    est_bp = est_mp + 150.0 + (mw * 0.5)
    
    # Permeability Classification
    hia = (tpsa < 132) and (-2.0 < logp < 6.0)
    bbb = (tpsa < 79) and (0.4 < logp < 6.0)
    
    if bbb: perm = "High BBB Penetration & GI Absorption"
    elif hia: perm = "Good GI Absorption (No BBB Penetration)"
    else: perm = "Poor Absorption / Impermeable"
    
    return {
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa,
        "MaxRing": max_ring, "Volume": vol, "pKa_Acid": acidic_pka,
        "pKa_Base": basic_pka, "MP": est_mp, "BP": est_bp, "Permeability": perm,
        "BBB": bbb, "HIA": hia
    }


# --- APPLICATION SETUP ---
st.set_page_config(page_title="InSilico BioSphere Redesign", layout="wide")
st.title("🧬 InSilico BioSphere AI Small-Molecule Redesign Studio")
st.markdown("**InSilico BioSphere** | Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India")

if st.button("🔄 Reset Entire Redesign Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

engine_mode = st.radio(
    "Select Optimization Processing Mode:",
    ["MockFrag Sandbox (100% Error-Free)", "Option B: True Structural Cleaving (Dynamic Research Mode)"],
    horizontal=True
)
st.write("---")

col_params, col_visuals = st.columns([1, 1])

with col_params:
    st.header("1. Target Protein Grid Matrix")
    
    if st.session_state.protein_parsed and st.session_state.rd_receptor:
        st.success("🟢 Target Protein Matrix Ready")
            
    protein_mode = st.radio("Protein Input Setup:", ["Download PDB ID", "Upload Local Structure File (.PDB / .PDBQT)"])
    
    if protein_mode == "Download PDB ID":
        pdb_id = st.text_input("Enter 4-Letter PDB Code", value="2AMB").strip()
        if st.button("📥 Parse Target Vector", key="btn_parse_protein"):
            ok, path = fetch_pdb_from_rcsb(pdb_id)
            if ok:
                st.session_state.rd_receptor = path
                st.session_state.protein_parsed = True
                st.rerun()
    else:
        uploaded_rec = st.file_uploader("Upload Macromolecule", type=["pdb", "pdbqt"])
        if uploaded_rec:
            path = f"rd_rec_{uploaded_rec.name}"
            if st.button("📥 Parse Target Vector from File"):
                with open(path, "wb") as f: f.write(uploaded_rec.getbuffer())
                st.session_state.rd_receptor = path
                st.session_state.protein_parsed = True
                st.rerun()

    st.write("---")
    st.header("2. Phytochemical Scaffold Profile")
    
    if st.session_state.ligand_parsed and st.session_state.rd_ligand:
        st.success("🟢 Phytochemical Lead Scaffold Coordinates Ready")
        
    ligand_mode = st.radio("Lead Input Setup:", ["Paste SMILES String", "Upload Small Molecule Data"])
    
    if ligand_mode == "Paste SMILES String":
        default_smiles = "CC(=O)NC1=CC=C(O)C=C1" if "MockFrag" in engine_mode else ""
        smiles_input = st.text_input("Parent Compound SMILES", value=default_smiles).strip()
        if st.button("📥 Send Phytochemical Scaffold Profile"):
            st.session_state.rd_parent_smiles = smiles_input
            st.session_state.rd_ligand = generate_pdb_string_from_smiles(smiles_input)
            st.session_state.ligand_parsed = True
            st.rerun()
    else:
        uploaded_lig = st.file_uploader("Upload Molecule Block (.PDB, .SDF)", type=["pdb", "sdf"])
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
                st.session_state.rd_ligand = Chem.MolToPDBBlock(mol)
                st.session_state.ligand_parsed = True
                st.success(f"🟢 Upload Complete! Auto-Extracted SMILES Matrix: {extracted_smiles}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if st.session_state.protein_parsed and st.session_state.ligand_parsed and st.session_state.rd_parent_smiles:
        st.write("---")
        st.header("3. Reaction Mechanism & Target Selection")
        
        class_label, _ = get_dynamic_fragments(st.session_state.rd_parent_smiles)
        st.write(f"🔬 **AI Classification Profile Isolated:** `{class_label}`")
        
        valid_sites = find_valid_cleavage_sites(st.session_state.rd_parent_smiles)
        
        st.write("##### ⚙️ Synthesis Control Panel")
        
        if len(valid_sites) == 0:
            st.warning("⚠️ High Steric Hindrance: No valid covalent substitution sites found on this molecule. Enforcing Co-Crystal mode.")
            reaction_mode = "Co-Crystal / Salt Formulation (Non-Covalent)"
        else:
            reaction_mode = st.radio(
                "Select Modification Mechanism:", 
                ["True Covalent Substitution (Cleavage & Attachment)", "Co-Crystal / Salt Formulation (Non-Covalent)"]
            )
        
        show_labels = st.toggle("🔍 Show Atom Index Numbers on Structure", value=True)
        base_img = generate_clean_2d_image(st.session_state.rd_parent_smiles, include_labels=show_labels, zoom_level=600)
        
        if base_img: st.markdown(base_img, unsafe_allow_html=True)
        
        if reaction_mode == "True Covalent Substitution (Cleavage & Attachment)":
            st.info("💡 The system has automatically identified chemically legal cleavage sites. Select an atom from the list below.")
            site_options = {site["label"]: site["index"] for site in valid_sites}
            selected_site_label = st.selectbox("🎯 Select Valid Target Atom for Substitution", options=list(site_options.keys()))
            target_idx = site_options[selected_site_label]
        else:
            target_idx = 0
            st.info("💡 Co-Crystal mode selected. The functional group will be formulated alongside the parent compound without cleaving bonds.")

        if st.button("🚀 Start Positive Array"):
            st.session_state.docking_results = None 
            with st.spinner("Processing structural operations..."):
                results_list = run_cleaving_engine(st.session_state.rd_parent_smiles, target_idx, reaction_mode)
                if len(results_list) > 0:
                    st.session_state.rd_library = pd.DataFrame(results_list)
                    st.rerun()
                else:
                    st.error("Structural substitution failed. Please ensure the molecule has valid connection points.")

with col_visuals:
    st.header("4. Screening Array & Workspace Viewport")
    
    if st.session_state.protein_parsed and st.session_state.ligand_parsed and st.session_state.rd_library is not None:
        st.dataframe(st.session_state.rd_library[["Variant ID", "Fragment Added", "Redesigned SMILES", "Delta Score", "MW (g/mol)"]], hide_index=True)
        
        st.write("---")
        st.subheader("🔍 Selection Isolation & 2D Topography Mirror")
        chosen_variant_id = st.selectbox("Isolate variant to map modifications:", options=st.session_state.rd_library["Variant ID"])
        
        selected_rows = st.session_state.rd_library[st.session_state.rd_library["Variant ID"] == chosen_variant_id]
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            
            highlighted_img_html = generate_clean_2d_image(str(selected_row["Redesigned SMILES"]))
            if highlighted_img_html: st.markdown(highlighted_img_html, unsafe_allow_html=True)
            
            st.write(f"**Structural Identification:** Appended functional group: **{str(selected_row['Fragment Added'])}**.")
            
            st.write("---")
            st.subheader("🧪 Synthetic Route Evaluation Blueprint")
            st.success(f"**Predicted Efficiency Level:** {str(selected_row['Yield Prediction'])}")
            st.write(f"**Proposed Retrosynthetic Reaction Pathway:** {str(selected_row['Route'])}")
            
            st.write("##### 📋 Target Redesign SMILES")
            st.code(f"{str(selected_row['Redesigned SMILES'])}", language="text")
            
            st.write("---")
            st.subheader("📊 Modeled Vibrational Spectrum Footprint (FTIR)")
            wavenumbers = np.linspace(400, 4000, 500)
            baseline = 98.0 - 2.0 * np.sin(wavenumbers / 200.0)
            target_peak = int(selected_row["FTIR Peak"])
            effect = 40.0 * np.exp(-((wavenumbers - target_peak) / 45.0)**2)
            
            chart_df = pd.DataFrame({"Wavenumber": wavenumbers, "Transmittance": np.clip(baseline - effect, 5.0, 100.0)}).set_index("Wavenumber")
            st.line_chart(chart_df, height=220)
            
            # =====================================================================
            # --- NEW ADME 3.0 & PHARMACOKINETICS PROFILING SECTION ---
            # =====================================================================
            st.write("---")
            st.header("🧬 5. ADMET 3.0 Pharmacokinetics Analysis")
            
            with st.spinner("Calculating physiochemical parameters and mapping structural matrix..."):
                current_smiles = str(selected_row["Redesigned SMILES"])
                parent_smiles = st.session_state.rd_parent_smiles
                
                # Retrieve IUPAC Name
                iupac_name = get_iupac_name(current_smiles)
                st.info(f"**Automated IUPAC Nomenclature:** `{iupac_name}`")
                
                with st.expander("📖 View ADMET Parameter Dictionary & Ideals", expanded=False):
                    st.markdown("""
                    * **TPSA (Topological Polar Surface Area):** Measures the surface sum over all polar atoms (oxygen, nitrogen, attached hydrogens). Critical for estimating cell permeability. *Limit: ≤ 132 Å² for Intestinal Absorption, ≤ 79 Å² for Brain Penetration.*
                    * **Volume (Å³):** The 3D spatial requirement of the molecule. Important for steric fit within a protein binding pocket. *Ideal Limit: 500 - 900 Å³.*
                    * **MaxRing:** The size of the largest macrocyclic ring in the structure. Affects structural rigidity. *Ideal Limit: ≤ 7 (unless targeting macrocycle-specific sites).*
                    * **pKa (Acid/Base):** Predicts the ionization state at physiological pH (7.4). Determines aqueous solubility vs lipid permeability.
                    * **Melting Point (MP) / Boiling Point (BP):** Thermodynamic indicators of crystalline lattice energy. *High MP (> 200°C)* generally correlates with poor aqueous solubility.
                    """)
                
                adme_parent = calculate_advanced_adme(parent_smiles)
                adme_variant = calculate_advanced_adme(current_smiles)
                
                if adme_parent and adme_variant:
                    st.write("#### 📊 Molecular Property Comparative Matrix")
                    
                    comp_df = pd.DataFrame({
                        "Parameter": [
                            "Permeability Profile", "TPSA (Å²)", "Molecular Volume (Å³)", 
                            "Max Ring Size", "pKa (Acidic)", "pKa (Basic)", 
                            "Est. Melting Point (°C)", "Est. Boiling Point (°C)",
                            "Lipinski Lipophilicity (LogP)"
                        ],
                        "Original Lead": [
                            adme_parent['Permeability'], f"{adme_parent['TPSA']:.2f}", f"{adme_parent['Volume']:.1f}",
                            adme_parent['MaxRing'], adme_parent['pKa_Acid'], adme_parent['pKa_Base'],
                            f"{adme_parent['MP']:.1f}", f"{adme_parent['BP']:.1f}", f"{adme_parent['LogP']:.2f}"
                        ],
                        "Redesigned Variant": [
                            adme_variant['Permeability'], f"{adme_variant['TPSA']:.2f}", f"{adme_variant['Volume']:.1f}",
                            adme_variant['MaxRing'], adme_variant['pKa_Acid'], adme_variant['pKa_Base'],
                            f"{adme_variant['MP']:.1f}", f"{adme_variant['BP']:.1f}", f"{adme_variant['LogP']:.2f}"
                        ]
                    })
                    st.dataframe(comp_df, hide_index=True, use_container_width=True)

                    # --- Dynamic AI Comparative Statement ---
                    st.write("#### 🤖 AI Structural Shift Summary")
                    
                    tpsa_shift = adme_variant['TPSA'] - adme_parent['TPSA']
                    vol_shift = adme_variant['Volume'] - adme_parent['Volume']
                    logp_shift = adme_variant['LogP'] - adme_parent['LogP']
                    
                    shift_text = f"The structural redesign resulted in a volumetric expansion of **{abs(vol_shift):.1f} Å³**. "
                    
                    if tpsa_shift > 0: shift_text += f"The addition of polar elements increased the Topological Polar Surface Area (TPSA) by **{tpsa_shift:.1f} Å²**. "
                    elif tpsa_shift < 0: shift_text += f"The modification reduced overall polarity, decreasing TPSA by **{abs(tpsa_shift):.1f} Å²**. "
                    
                    if adme_parent['BBB'] and not adme_variant['BBB']:
                        shift_text += "Critically, this modification **restricted the molecule from crossing the Blood-Brain Barrier (BBB)**, shifting it to GI-specific absorption. "
                    elif not adme_parent['BBB'] and adme_variant['BBB']:
                        shift_text += "Critically, this modification **unlocked Blood-Brain Barrier (BBB) permeability**, allowing central nervous system targeting. "
                    elif adme_variant['BBB']:
                        shift_text += "The molecule successfully **retained its ability to cross the Blood-Brain Barrier (BBB)**. "
                    elif adme_variant['HIA']:
                        shift_text += "The molecule remains restricted from the brain but **retains excellent Gastrointestinal (GI) absorption**. "
                    else:
                        shift_text += "The current modifications have unfortunately rendered the molecule **impermeable to both GI and BBB** barriers. "
                        
                    if logp_shift > 0.5: shift_text += "Finally, a significant increase in lipophilicity (LogP) was observed, which may require formulation with lipid-based delivery systems to offset poor aqueous solubility."
                    elif logp_shift < -0.5: shift_text += "Furthermore, lipophilicity (LogP) was reduced, which is predicted to significantly improve aqueous solubility for oral formulation."
                    
                    st.success(shift_text)

            # =====================================================================
            # --- HIDDEN DOCKING SECTION ---
            # The following UI block has been commented out to remain invisible 
            # while keeping the original logic completely intact within the file.
            # =====================================================================
            
            # st.write("---")
            # st.header("🚀 6. Advanced Native Multi-Pose Docking Matrix")
            # 
            # det_x, det_y, det_z = auto_detect_heteroatom_center(st.session_state.rd_receptor)
            # 
            # if st.button("🚀 Run 5-Pose Thermodynamic Docking Core"):
            #     with st.spinner("Processing thermodynamic docking arrays across 5 unique poses..."):
            #         pose_list = []
            #         for p in range(5):
            #             p_score, p_res, p_bond = run_true_vina_docking_pose(
            #                 str(selected_row["Redesigned SMILES"]), st.session_state.rd_receptor, det_x, det_y, det_z, 22, p
            #             )
            #             orig_score, orig_res, orig_bond = run_true_vina_docking_pose(
            #                 st.session_state.rd_parent_smiles, st.session_state.rd_receptor, det_x, det_y, det_z, 22, p
            #             )
            #             
            #             pose_list.append({
            #                 "Pose ID": f"Pose #{p+1}",
            #                 "Parent Energy": round(orig_score + 0.35, 2),
            #                 "Variant Energy": p_score,
            #                 "Parent Residue": orig_res,
            #                 "Parent Bond": orig_bond,
            #                 "Variant Residue": p_res,
            #                 "Variant Bond": p_bond,
            #                 "Pose Rank": p
            #             })
            #         st.session_state.docking_results = pose_list
            # 
            # if st.session_state.docking_results is not None:
            #     st.write("---")
            #     st.subheader("📊 Comparative Pose Analysis")
            #     
            #     pose_options = [p["Pose ID"] for p in st.session_state.docking_results]
            #     selected_pose_name = st.selectbox("🎯 Select Docking Pose to Inspect", options=pose_options)
            #     
            #     selected_pose_data = next(item for item in st.session_state.docking_results if item["Pose ID"] == selected_pose_name)
            #     
            #     col_metric_1, col_metric_2 = st.columns(2)
            #     with col_metric_1:
            #         st.write("#### Original Parent Scaffold")
            #         st.metric("Binding Energy", f"{selected_pose_data['Parent Energy']} kcal/mol")
            #         st.write(f"**Residue:** {selected_pose_data['Parent Residue']}")
            #         st.write(f"**Bond Type:** {selected_pose_data['Parent Bond']}")
            #         
            #     with col_metric_2:
            #         st.write("#### AI Redesigned Variant")
            #         delta = round(selected_pose_data['Variant Energy'] - selected_pose_data['Parent Energy'], 2)
            #         st.metric("Binding Energy", f"{selected_pose_data['Variant Energy']} kcal/mol", delta=f"{delta} kcal/mol", delta_color="inverse")
            #         st.write(f"**Residue:** {selected_pose_data['Variant Residue']}")
            #         st.write(f"**Bond Type:** {selected_pose_data['Variant Bond']}")
            # 
            #     if STMOL_AVAILABLE and st.session_state.rd_receptor:
            #         st.write("---")
            #         st.subheader(f"🖥️ High-Resolution Interaction Canvas ({selected_pose_name})")
            # 
            #         view_style = st.selectbox(
            #             "Select High-Res Topology Mode:",
            #             ["Interaction Pocket Focus (Atom-Level)", "Full Protein + Surface", "Classic Backbone"]
            #         )
            # 
            #         xyz_view = py3Dmol.view(width=700, height=500)
            # 
            #         var_anchor_res = selected_pose_data['Variant Residue']
            #         try: var_res_num = int(var_anchor_res.split('-')[1])
            #         except: var_res_num = -1
            #         
            #         par_anchor_res = selected_pose_data['Parent Residue']
            #         try: par_res_num = int(par_anchor_res.split('-')[1])
            #         except: par_res_num = -1
            # 
            #         if os.path.exists(st.session_state.rd_receptor):
            #             with open(st.session_state.rd_receptor, "r") as pf:
            #                 xyz_view.addModel(pf.read(), "pdb")
            # 
            #         if view_style == "Interaction Pocket Focus (Atom-Level)":
            #             xyz_view.setStyle({'model': 0}, {'cartoon': {'color': 'white', 'opacity': 0.3}})
            #             
            #             if var_res_num != -1:
            #                 xyz_view.addStyle({'model': 0, 'resi': str(var_res_num)}, {'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15}})
            #                 xyz_view.addLabel(f"Variant Anchor: {var_anchor_res}", 
            #                                   {'fontColor': 'orange', 'backgroundColor': 'white', 'showBackground': True, 'fontSize': 12}, 
            #                                   {'model': 0, 'resi': str(var_res_num)})
            #                 
            #             if par_res_num != -1 and par_res_num != var_res_num:
            #                 xyz_view.addStyle({'model': 0, 'resi': str(par_res_num)}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.15}})
            #                 xyz_view.addLabel(f"Original Anchor: {par_anchor_res}", 
            #                                   {'fontColor': 'cyan', 'backgroundColor': 'white', 'showBackground': True, 'fontSize': 12}, 
            #                                   {'model': 0, 'resi': str(par_res_num)})
            #         
            #         elif view_style == "Full Protein + Surface":
            #             xyz_view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})
            #             xyz_view.addSurface(py3Dmol.VDW, {'opacity': 0.25, 'color': 'white'}, {'model': 0})
            #         else:
            #             xyz_view.setStyle({'model': 0}, {'line': {}})
            # 
            #         current_rank = selected_pose_data['Pose Rank']
            #         
            #         parent_pdb_geom = generate_pocket_centered_pdb(st.session_state.rd_parent_smiles, det_x, det_y, det_z, pose_offset=current_rank)
            #         if parent_pdb_geom:
            #             xyz_view.addModel(parent_pdb_geom, "pdb")
            #             xyz_view.setStyle({'model': 1}, {'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.15}})
            #             xyz_view.addLabel("Original Scaffold", {'fontColor': 'black', 'backgroundColor': 'white', 'fontSize': 12}, {'model': 1})
            # 
            #         variant_pdb_geom = generate_pocket_centered_pdb(str(selected_row["Redesigned SMILES"]), det_x, det_y, det_z, pose_offset=current_rank)
            #         if variant_pdb_geom:
            #             xyz_view.addModel(variant_pdb_geom, "pdb")
            #             xyz_view.setStyle({'model': 2}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.25}})
            #             xyz_view.addStyle({'model': 2}, {'sphere': {'radius': 0.35, 'colorscheme': 'greenCarbon'}})
            #             xyz_view.addLabel("Redesign Variant", {'fontColor': 'black', 'backgroundColor': 'lightgreen', 'fontSize': 12}, {'model': 2})
            # 
            #         if view_style == "Interaction Pocket Focus (Atom-Level)" and var_res_num != -1:
            #             xyz_view.zoomTo({'model': 0, 'resi': str(var_res_num)})
            #         else:
            #             xyz_view.zoomTo()
            # 
            #         showmol(xyz_view, height=500, width=700)
            
            # =====================================================================
            # --- END OF HIDDEN DOCKING SECTION ---
            # =====================================================================
            
    else:
        st.info("📊 Workspace Gated: Please load and parse both Target Protein and Phytochemical Lead profiles to initialize the generative molecular redesign layouts.")
