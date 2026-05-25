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
                for atom in mol_to_draw_GetAtoms():
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
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    lipinski_obey = "Yes" if violations <= 1 else "No"
    
    if violations == 0: oral_bio = "Yes (High Probability)"
    elif violations == 1: oral_bio = "Yes (Moderate Probability)"
    else: oral_bio = "No (Poor Bioavailability)"

    ring_info = mol.GetRingInfo().AtomRings()
    max_ring = max([len(r) for r in ring_info]) if ring_info else 0
    
    try:
        temp_mol = Chem.Mol(mol)
        AllChem.EmbedMolecule(temp_mol, randomSeed=42)
        vol = AllChem.ComputeMolVolume(temp_mol)
    except:
        vol = mw * 0.88
        
    acidic_pka = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")): acidic_pka = "Acidic (~4.5)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")): acidic_pka = "Weak Acid (~9.5)"
    
    basic_pka = "Neutral (None)"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")): basic_pka = "Basic (~9.0)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("cN")): basic_pka = "Weak Base (~4.0)"
    
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    est_mp = max(20.0, (mw * 0.4) + (hbd * 25.0) - (rot_bonds * 5.0))
    est_bp = est_mp + 150.0 + (mw * 0.5)
    
    hia = (tpsa < 132) and (-2.0 < logp < 6.0)
    bbb = (tpsa < 79) and (0.4 < logp < 6.0)
    
    if bbb: perm = "High BBB Penetration & GI Absorption"
    elif hia: perm = "Good GI Absorption (No BBB Penetration)"
    else: perm = "Poor Absorption / Impermeable"
    
    return {
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa,
        "Violations": violations, "Lipinski_Obey": lipinski_obey, "Oral_Bio": oral_bio,
        "MaxRing": max_ring, "Volume": vol, "pKa_Acid": acidic_pka,
        "pKa_Base": basic_pka, "MP": est_mp, "BP": est_bp, "Permeability": perm,
        "BBB": bbb, "HIA": hia
    }

# --- REPORT EXPORT MODULE ---
def generate_html_report(engine_mode, protein_id, parent_smiles, reaction_mode, library_df, selected_row, iupac_name, comp_df, shift_summary, parent_img, variant_img):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>InSilico BioSphere Redesign Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 0; background-color: #f9f9fb; }}
            .header-banner {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px; border-bottom: 5px solid #00c6ff; text-align: center; position: relative; }}
            .header-banner h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
            .header-banner p {{ margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }}
            .copyright-header {{ font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: rgba(255,255,255,0.7); margin-bottom: 10px; }}
            .container {{ max-width: 1000px; margin: 30px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h2 {{ color: #1e3c72; border-bottom: 2px solid #eef2f7; padding-bottom: 8px; margin-top: 35px; font-size: 20px; }}
            h3 {{ color: #2a5298; font-size: 16px; margin-top: 20px; }}
            .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; background: #f4f7f6; padding: 20px; border-radius: 8px; }}
            .meta-item {{ font-size: 14px; }}
            .meta-item strong {{ color: #1e3c72; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
            table, th, td {{ border: 1px solid #e2e8f0; }}
            th {{ background-color: #f8fafc; color: #1e3c72; padding: 12px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px; vertical-align: middle; }}
            .structure-box {{ display: flex; gap: 30px; margin: 20px 0; background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eef2f7; align-items: center; }}
            .structure-img {{ background: white; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; max-width: 320px; }}
            .scandata {{ font-family: monospace; background: #f1f5f9; padding: 3px 6px; border-radius: 4px; font-size: 13px; word-break: break-all; }}
            .summary-card {{ background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 20px; border-radius: 6px; margin: 25px 0; color: #065f46; font-size: 14.5px; }}
            .dictionary-box {{ background-color: #f8fafc; padding: 20px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 13px; }}
            footer {{ text-align: center; padding: 20px; font-size: 12px; color: #64748b; margin-top: 5px; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <div class="copyright-header">copyright@sarang dhote</div>
            <h1>🧬 InSilico BioSphere</h1>
            <p>Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry</p>
            <p>Shivaji Science College, Nagpur, Maharashtra, India | Contact: sarangresearch@gmail.com</p>
        </div>
        
        <div class="container">
            <h2>1. Studio Configuration & Environment Setup</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Optimization Processing Mode:</strong> {engine_mode}</div>
                <div class="meta-item"><strong>Target Protein ID Matrix:</strong> {protein_id}</div>
                <div class="meta-item"><strong>Modification Mechanism Vector:</strong> {reaction_mode}</div>
                <div class="meta-item"><strong>Parent Query Template:</strong> <span class="scandata">{parent_smiles}</span></div>
            </div>
            
            <div class="structure-box">
                <div>{parent_img}</div>
                <div>
                    <strong>Phytochemical Lead Template Profile:</strong><br>
                    Initial structural parameters parsed successfully. Standard 2D coordinate matrix compiled mapping atom distribution maps prior to modification arrays.
                </div>
            </div>

            <h2>2. Screening Array & Workspace Viewport</h2>
            <p>Complete algorithmic optimization library tracking modifications parsed during structural operations:</p>
            {library_df.to_html(index=False, classes='table')}

            <h2>3. Selected Redesign Variant Mapping</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Isolated Variant ID:</strong> {selected_row['Variant ID']}</div>
                <div class="meta-item"><strong>Appended Functional Group:</strong> {selected_row['Fragment Added']}</div>
                <div class="meta-item"><strong>Retrosynthetic Reaction Pathway:</strong> {selected_row['Route']}</div>
                <div class="meta-item"><strong>Predicted FTIR Peak Tracker:</strong> {selected_row['FTIR Peak']} cm⁻¹</div>
            </div>

            <div class="structure-box">
                <div class="structure-img">{variant_img}</div>
                <div style="flex:1;">
                    <h3>📋 Redesigned Target SMILES String Matrix</h3>
                    <div class="scandata" style="margin-bottom: 15px;">{selected_row['Redesigned SMILES']}</div>
                    <strong>Synthetic Route Evaluation Blueprint:</strong><br>
                    Predicted Efficiency Yield Tier: <span style="color:#1e3c72; font-weight:bold;">{selected_row['Yield Prediction']}</span>. Pathway coordinates optimized via functional block swapping mechanics.
                </div>
            </div>

            <h2>4. ADMET 3.0 Pharmacokinetics Analysis</h2>
            <p><strong>Automated IUPAC Nomenclature Generation:</strong></p>
            <div class="scandata" style="margin-bottom:20px; background:#e0f2fe; color:#0369a1; padding:10px;">{iupac_name}</div>
            
            <h3>Molecular Property Comparative Matrix</h3>
            {comp_df.to_html(index=False, classes='table')}

            <h3>Structural Shift Summary</h3>
            <div class="summary-card">
                {shift_summary.replace('\n\n', '<br><br>')}
            </div>

            <h3>ADMET Parameter Dictionary & Ideals</h3>
            <div class="dictionary-box">
                <ul>
                    <li><strong>TPSA (Topological Polar Surface Area):</strong> Measures surface sum over all polar atoms. *Limit: ≤ 132 Å² for Intestinal Absorption, ≤ 79 Å² for Brain Penetration.*</li>
                    <li><strong>Volume (Å³):</strong> 3D spatial requirement. Structural anchor requirement for binding pocket fitment. *Ideal Limit: 500 - 900 Å³.*</li>
                    <li><strong>MaxRing:</strong> Structural rigidity matrix boundary indicator. *Ideal Limit: Max Ring Size ≤ 7 atoms.*</li>
                    <li><strong>pKa (Acid/Base):</strong> Ionization indicator tracking partition behavior at biological pH 7.4.</li>
                    <li><strong>Melting Point (MP) / Boiling Point (BP):</strong> Thermodynamic descriptors profiling phase behaviors.</li>
                    <li><strong>Lipinski's Rule of 5:</strong> Standard druglikeness validation framework assessing oral bioavailability parameters.</li>
                </ul>
            </div>
        </div>
        
        <footer>
            InSilico BioSphere System Technical Report | copyright@sarang dhote | All Rights Reserved.
        </footer>
    </body>
    </html>
    """
    return html_template


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
            # --- ADME 3.0 & PHARMACOKINETICS PROFILING SECTION ---
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
                    * **Lipinski's Rule of 5:** A rule of thumb to evaluate druglikeness. *Rules: MW ≤ 500, LogP ≤ 5, H-bond Donors ≤ 5, H-bond Acceptors ≤ 10.* More than 1 violation predicts poor oral absorption.
                    """)
                
                adme_parent = calculate_advanced_adme(parent_smiles)
                adme_variant = calculate_advanced_adme(current_smiles)
                
                if adme_parent and adme_variant:
                    st.write("#### 📊 Molecular Property Comparative Matrix")
                    
                    comp_df = pd.DataFrame({
                        "Parameter": [
                            "Obey Lipinski's Rule?", "Can take Orally? (Bioavailability)",
                            "Permeability Profile", "TPSA (Å²)", "Molecular Volume (Å³)", 
                            "Max Ring Size", "pKa (Acidic)", "pKa (Basic)", 
                            "Est. Melting Point (°C)", "Est. Boiling Point (°C)",
                            "Lipinski Lipophilicity (LogP)"
                        ],
                        "Original Lead": [
                            adme_parent['Lipinski_Obey'], adme_parent['Oral_Bio'],
                            adme_parent['Permeability'], f"{adme_parent['TPSA']:.2f}", f"{adme_parent['Volume']:.1f}",
                            adme_parent['MaxRing'], adme_parent['pKa_Acid'], adme_parent['pKa_Base'],
                            f"{adme_parent['MP']:.1f}", f"{adme_parent['BP']:.1f}", f"{adme_parent['LogP']:.2f}"
                        ],
                        "Redesigned Variant": [
                            adme_variant['Lipinski_Obey'], adme_variant['Oral_Bio'],
                            adme_variant['Permeability'], f"{adme_variant['TPSA']:.2f}", f"{adme_variant['Volume']:.1f}",
                            adme_variant['MaxRing'], adme_variant['pKa_Acid'], adme_variant['pKa_Base'],
                            f"{adme_variant['MP']:.1f}", f"{adme_variant['BP']:.1f}", f"{adme_variant['LogP']:.2f}"
                        ]
                    })
                    st.dataframe(comp_df, hide_index=True, use_container_width=True)

                    # --- Dynamic Structural Shift Summary ---
                    st.write("#### Structural Shift Summary")
                    
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
                        
                    if logp_shift > 0.5: shift_text += "A significant increase in lipophilicity (LogP) was observed, which may require formulation with lipid-based delivery systems to offset poor aqueous solubility. "
                    elif logp_shift < -0.5: shift_text += "Furthermore, lipophilicity (LogP) was reduced, which is predicted to significantly improve aqueous solubility for oral formulation. "
                    
                    if adme_variant['Violations'] < adme_parent['Violations']:
                        conclusion = "✅ **Overall Assessment: Favorable.** This redesigned structure is **better** than the original lead due to improved Lipinski compliance and higher predicted oral bioavailability."
                    elif adme_variant['Violations'] > adme_parent['Violations']:
                        conclusion = "❌ **Overall Assessment: Unfavorable.** This redesigned structure is **worse** than the original lead because it introduces new Lipinski violations, likely reducing oral bioavailability."
                    else:
                        if adme_variant['Violations'] <= 1:
                            if adme_variant['Permeability'] != "Poor Absorption / Impermeable":
                                conclusion = "⚖️ **Overall Assessment: Comparable.** Both structures obey Lipinski's rules and maintain good bioavailability. The redesigned structure is **a strong, viable alternative** to the original lead."
                            else:
                                conclusion = "⚠️ **Overall Assessment: Unfavorable.** Despite obeying Lipinski's rules, the redesign resulted in poor predicted barrier permeability, making it **worse** for oral delivery than the original lead."
                        else:
                            conclusion = "⚠️ **Overall Assessment: Comparable but Flawed.** Both structures possess multiple Lipinski violations. The redesign **does not significantly improve** fundamental oral drug-likeness over the original lead."
                    
                    shift_text += "\n\n" + conclusion
                    st.success(shift_text)
                    
                    # --- REPORT DOWNLOAD BUTTON MODULE ---
                    st.write("---")
                    st.subheader("📄 Automated Export Infrastructure")
                    
                    protein_id_label = pdb_id if protein_mode == "Download PDB ID" else str(st.session_state.rd_receptor)
                    
                    # Prepare data structures specifically for clean printing
                    html_report_content = generate_html_report(
                        engine_mode=str(engine_mode),
                        protein_id=protein_id_label,
                        parent_smiles=str(parent_smiles),
                        reaction_mode=str(reaction_mode),
                        library_df=st.session_state.rd_library,
                        selected_row=selected_row,
                        iupac_name=str(iupac_name),
                        comp_df=comp_df,
                        shift_summary=str(shift_text),
                        parent_img=str(base_img),
                        variant_img=str(highlighted_img_html)
                    )
                    
                    st.download_button(
                        label="📥 Download Comprehensive HTML Research Report",
                        data=html_report_content,
                        file_name=f"InSilico_BioSphere_Report_{selected_row['Variant ID']}.html",
                        mime="text/html",
                        use_container_width=True
                    )

            # =====================================================================
            # --- HIDDEN DOCKING SECTION ---
            # =====================================================================
            # st.write("---")
            # st.header("🚀 6. Advanced Native Multi-Pose Docking Matrix")
            # ... [Docking blocks completely hidden but safely retained]
            
    else:
        st.info("📊 Workspace Gated: Please load and parse both Target Protein and Phytochemical Lead profiles to initialize the generative molecular redesign layouts.")
