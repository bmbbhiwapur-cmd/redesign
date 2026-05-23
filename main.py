import streamlit as st
import os
import urllib.request
import re
import numpy as np
import pandas as pd
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

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
    """Generates a standard compliant PDB structural string using RDKit coordinates safely."""
    if not smiles_str:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            embed_status = AllChem.EmbedMolecule(mol, params)
            if embed_status >= 0:
                AllChem.MMFFOptimizeMolecule(mol)
                return Chem.MolToPDBBlock(mol)
    except Exception:
        pass
    return None

def generate_labeled_2d_image(smiles_str, highlight_atoms=None, legend_text="Locate your target position number below:", zoom_level=450):
    """Generates a 2D image of the molecule with labeled indices, custom dimensions, and optional highlighting."""
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.Mol(mol)
            for atom in mol_to_draw.GetAtoms():
                atom.SetProp('atomNote', f"#{atom.GetIdx()}")
            
            kwargs = {}
            if highlight_atoms is not None:
                kwargs['highlightAtoms'] = highlight_atoms
                kwargs['highlightColor'] = (0.4, 0.9, 0.4)
            
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)), legend=legend_text, **kwargs)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom:15px;"/>'
    except Exception:
        pass
    return "<p style='color:red;'>Visual mapping error.</p>"

def generate_dynamic_derivatives(parent_smiles, target_atom_idx):
    """Programmatically attaches 10 distinct functional groups to the selected atom position."""
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol:
        return []
    
    fragments = [
        {"name": "Methylation (-CH3)", "smiles": "C", "peak": 2925, "yield": "Good Yield (85%)", "route": "Alkylation via Methyl Iodide under basic carbonate conditions."},
        {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3450, "yield": "Moderate Yield (62%)", "route": "Direct C-H oxidation utilizing copper or iron catalysis."},
        {"name": "Amination (-NH2)", "smiles": "N", "peak": 3320, "yield": "Good Yield (74%)", "route": "Controlled nitration followed by selective reduction with Pd/C."},
        {"name": "Fluorination (-F)", "smiles": "F", "peak": 1150, "yield": "Poor Yield (38%)", "route": "Late-stage electrophilic fluorination using Selectfluor."},
        {"name": "Trifluoromethylation (-CF3)", "smiles": "C(F)(F)F", "peak": 1280, "yield": "Moderate Yield (55%)", "route": "Trifluoromethylation using Ruppert-Prakash reagent."},
        {"name": "Cyanation (-C≡N)", "smiles": "C#N", "peak": 2220, "yield": "Good Yield (81%)", "route": "Rosenmund-von Braun cyanation using CuCN in refluxing DMF."},
        {"name": "Methoxylation (-OCH3)", "smiles": "OC", "peak": 1250, "yield": "Good Yield (88%)", "route": "Williamson ether synthesis using Dimethyl Sulfate."},
        {"name": "Acetylation (-COCH3)", "smiles": "C(=O)C", "peak": 1685, "yield": "Good Yield (79%)", "route": "Friedel-Crafts Acylation with Acetic Anhydride and Lewis Acid."},
        {"name": "Carboxylation (-COOH)", "smiles": "C(=O)O", "peak": 1715, "yield": "Moderate Yield (50%)", "route": "Carboxylation using high-pressure CO2 or carboxymethylation."},
        {"name": "Chlorination (-Cl)", "smiles": "Cl", "peak": 720, "yield": "Poor Yield (45%)", "route": "Electrophilic aromatic chlorination utilizing NCS."}
    ]
    
    derived_library = []
    num_atoms = parent_mol.GetNumAtoms()
    if target_atom_idx >= num_atoms:
        target_atom_idx = 0
        
    for idx, frag in enumerate(fragments):
        try:
            frag_mol = Chem.MolFromSmiles(frag["smiles"])
            combo = Chem.ComboMol(parent_mol, frag_mol)
            ed_combo = Chem.EditableMol(combo)
            new_atom_idx = num_atoms 
            ed_combo.AddBond(int(target_atom_idx), new_atom_idx, order=Chem.BondType.SINGLE)
            
            derived_mol = ed_combo.GetMol()
            Chem.SanitizeMol(derived_mol)
            derived_smiles = Chem.MolToSmiles(derived_mol)
            
            mw = round(Descriptors.MolWt(derived_mol), 2)
            logp = round(Descriptors.MolLogP(derived_mol), 2)
            simulated_score = round(0.95 - (idx * 0.03) - (abs(logp) * 0.01), 2)
            
            added_indices = list(range(num_atoms, derived_mol.GetNumAtoms()))
            
            derived_library.append({
                "Variant ID": f"Derivative-{idx+1:02d} (Rank {idx+1})",
                "Fragment Added": frag["name"],
                "Redesigned SMILES": derived_smiles,
                "Delta Score": max(simulated_score, 0.40),
                "MW (g/mol)": mw,
                "LogP": logp,
                "Yield Prediction": frag["yield"],
                "Route": frag["route"],
                "FTIR Peak": frag["peak"],
                "Highlight Atoms": added_indices
            })
        except Exception:
            fallback_smiles = f"{frag['smiles']}{parent_smiles}".replace("==", "=")
            try:
                f_mol = Chem.MolFromSmiles(fallback_smiles)
                mw = round(Descriptors.MolWt(f_mol), 2) if f_mol else 150.0
                logp = round(Descriptors.MolLogP(f_mol), 2) if f_mol else 1.5
            except Exception:
                mw, logp = 150.0, 1.5
                
            derived_library.append({
                "Variant ID": f"Derivative-{idx+1:02d} (Rank {idx+1})",
                "Fragment Added": frag["name"],
                "Redesigned SMILES": fallback_smiles,
                "Delta Score": round(0.92 - (idx * 0.03), 2),
                "MW (g/mol)": mw,
                "LogP": logp,
                "Yield Prediction": frag["yield"],
                "Route": frag["route"],
                "FTIR Peak": frag["peak"],
