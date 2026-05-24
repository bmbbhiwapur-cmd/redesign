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

# --- DEPENDENCY CHECKS ---
try:
    import py3Dmol
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError: STMOL_AVAILABLE = False

try:
    from vina import Vina
    from meeko import MoleculePreparation
    VINA_AVAILABLE = True
except ImportError: VINA_AVAILABLE = False

try:
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError: OPENBABEL_AVAILABLE = False

# --- SESSION INITIALIZATION ---
def initialize_session():
    defaults = {
        "rd_receptor": None, "temp_pdb_path": None, "rd_ligand": None, 
        "rd_parent_smiles": None, "rd_library": None, "docking_results": None,
        "protein_parsed": False, "ligand_parsed": False, "vina_poses_pdbqt": None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session()

# --- PREPARATION FUNCTIONS ---
def prepare_receptor_to_pdbqt(input_pdb_path):
    if not OPENBABEL_AVAILABLE: return False, "OpenBabel not installed. Manual conversion required."
    output_pdbqt = input_pdb_path.replace(".pdb", ".pdbqt")
    try:
        mols = list(pybel.readfile("pdb", input_pdb_path))
        if not mols: return False, "Failed to read PDB file."
        mol = mols[0]
        mol.removeh()
        mol.addh()
        mol.calccharges("gasteiger")
        mol.write("pdbqt", output_pdbqt, overwrite=True)
        return True, output_pdbqt
    except Exception as e: return False, str(e)

# --- (Keep all other helper functions: generate_pocket_centered_pdb, run_strict_vina_docking, etc. from previous version) ---

# --- APPLICATION UI ---
st.set_page_config(page_title="InSilico BioSphere Redesign", layout="wide")
st.title("🧬 InSilico BioSphere AI")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Receptor Setup")
    
    # Always allow upload
    uploaded_file = st.file_uploader("Upload PDB or PDBQT", type=["pdb", "pdbqt"])
    if uploaded_file:
        path = f"temp_{uploaded_file.name}"
        if st.session_state.temp_pdb_path != path:
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.temp_pdb_path = path
            st.rerun()

    if st.session_state.temp_pdb_path:
        st.info(f"File: {os.path.basename(st.session_state.temp_pdb_path)}")
        
        if st.session_state.temp_pdb_path.endswith(".pdbqt"):
            if st.button("✅ Load PDBQT File"):
                st.session_state.rd_receptor = st.session_state.temp_pdb_path
                st.session_state.protein_parsed = True
                st.rerun()
        else:
            if OPENBABEL_AVAILABLE:
                if st.button("⚙️ Convert PDB to PDBQT & Load"):
                    success, final_path = prepare_receptor_to_pdbqt(st.session_state.temp_pdb_path)
                    if success:
                        st.session_state.rd_receptor = final_path
                        st.session_state.protein_parsed = True
                        st.rerun()
                    else: st.error(final_path)
            else:
                st.warning("Auto-conversion unavailable. Please manually convert your PDB to PDBQT using AutoDock Tools and upload the .pdbqt file.")

with col2:
    st.header("2. Ligand Setup")
    smiles = st.text_input("Parent SMILES", "CC(=O)NC1=CC=C(O)C=C1")
    if st.button("Parse Ligand"):
        st.session_state.rd_parent_smiles = smiles
        st.session_state.ligand_parsed = True
        st.rerun()

# ... (Continue with the rest of your Docking & Viewer logic as before)
