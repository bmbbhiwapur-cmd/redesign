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

def compute_adme_profile(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1

    hia_absorbent = tpsa < 132 and logp < 5.88
    bbb_permeant = tpsa < 79 and 0.4 <= logp <= 6.0

    if bbb_permeant:
        egg_zone = "Yolk (High BBB Permeability & GI Absorption)"
        egg_color = "🍳"
    elif hia_absorbent:
        egg_zone = "White (High GI Absorption, No BBB Cross)"
        egg_color = "🥚"
    else:
        egg_zone = "Outside (Poor Absorption Profile)"
        egg_color = "🚫"

    return {
        "MW": round(mw, 2), "LogP": round(logp, 2), "HBD": hbd, "HBA": hba,
        "TPSA": round(tpsa, 2), "Rotatable Bonds": rot_bonds,
        "Lipinski Violations": violations, "Egg Zone": egg_zone, "Egg Color": egg_color
    }

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
st.set_page_config(page_title="InSilico BioSphere Red
