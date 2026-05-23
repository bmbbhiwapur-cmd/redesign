import streamlit as st
import os
import urllib.request
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

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

def render_comparison_viewport(parent_pdb, variant_pdb):
    """Uses 3Dmol.js to display a dual side-by-side interactive canvas comparing modifications."""
    import streamlit.components.v1 as components
    safe_parent = parent_pdb.replace('`', '\\`').replace('$', '\\$') if parent_pdb else ""
    safe_variant = variant_pdb.replace('`', '\\`').replace('$', '\\$') if variant_pdb else ""

    html_content = f"""
    <div style="display: flex; gap: 10px; width: 100%;">
        <div style="flex: 1;">
            <div style="text-align: center; font-weight: bold; font-family: sans-serif; margin-bottom: 5px; font-size: 14px; color: #555;">Original Scaffold Profile</div>
            <div id="container_parent" style="height: 320px; border: 1px solid #eaeaea; border-radius: 8px; background: #ffffff;"></div>
        </div>
        <div style="flex: 1;">
            <div style="text-align: center; font-weight: bold; font-family: sans-serif; margin-bottom: 5px; font-size: 14px; color: #2e7d32;">AI Redesigned Variant</div>
            <div id="container_variant" style="height: 320px; border: 1px solid #eaeaea; border-radius: 8px; background: #ffffff;"></div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            let parentData = `{safe_parent}`.trim();
            let variantData = `{safe_variant}`.trim();

            if (parentData.length > 0) {{
                let v_parent = $3Dmol.createViewer(document.getElementById('container_parent'), {{backgroundColor: '#ffffff'}});
                v_parent.addModel(parentData, 'pdb');
                v_parent.setStyle({{}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.25}}}});
                v_parent.zoomTo(); v_parent.render();
            }}

            if (variantData.length > 0) {{
                let v_variant = $3Dmol.createViewer(document.getElementById('container_variant'), {{backgroundColor: '#ffffff'}});
                v_variant.addModel(variantData, 'pdb');
                v_variant.setStyle({{}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.25}}}});
                v_variant.zoomTo(); v_variant.render();
            }}
        }});
    </script>
    """
    components.html(html_content, height=350)

# --- APPLICATION SETUP ---
st.set_page_config(page_title="InSilico BioSphere Redesign", layout="wide")
st.title("🧬 InSilico BioSphere AI Small-Molecule Redesign Studio")
st.markdown("""
**InSilico BioSphere** | Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
""")

# Initialize state trackers safely
if "rd_receptor" not in st.session_state: st.session_state.rd_receptor = None
if "rd_ligand" not in st.session_state: st.session_state.rd_ligand = None
if "rd_parent_smiles" not in st.session_state: st.session_state.rd_parent_smiles = ""
if "rd_library" not in st.session_state: st.session_state.rd_library = None

# --- MASTER ENVIRONMENT RESET ACTIONS ---
if st.button("🔄 Reset Entire Redesign Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Redesign parameters completely cleared!")
    st.rerun()

col_inputs, col_visuals = st.columns([1, 1])

with col_inputs:
    st.header("1. Target Protein Grid Matrix")
    protein_mode = st.radio("Protein Input Setup:", ["Download PDB ID", "Upload Local Structure File"])
    
    if protein_mode == "Download PDB ID":
        pdb_id = st.text_input("Enter 4-Letter P
