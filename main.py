# --- NEW USER-FRIENDLY POSITION MAPPING SYSTEM ---
    if st.session_state.rd_ligand is not None:
        st.write("---")
        st.header("3. Generative Growth Position")
        st.markdown("Select the specific chemical region of your molecule you want the AI to redesign:")

        # 1. Define user-friendly descriptions mapped directly to standard indices
        # We can dynamically check if the default Acetaminophen is being used to show customized labels
        is_default = "CC(=O)NC1=CC=C(O)C=C1" in st.session_state.rd_parent_smiles
        
        if is_default:
            position_options = {
                "Aromatic Ring: Position C-2 (Ortho to Hydroxyl)": 0,
                "Aromatic Ring: Position C-3 (Meta to Hydroxyl)": 1,
                "Phenolic Group: Oxygen Center (-OH root)": 4,
                "Amide Group: Nitrogen Center (-NH- bridge)": 7,
                "Carbonyl Link: Carbon Center (C=O)": 5,
                "Aliphatic Tail: Terminal Methyl Carbon (-CH3)": 6
            }
        else:
            # Smart fallback array for custom uploaded structures
            parent_mol = Chem.MolFromSmiles(st.session_state.rd_parent_smiles)
            total_atoms = parent_mol.GetNumAtoms() if parent_mol else 10
            position_options = {f"Atom Position Index #{i} ({parent_mol.GetAtomWithIdx(i).GetSymbol() if parent_mol else 'C'})": i for i in range(total_atoms)}

        # 2. Render a clean dropdown menu instead of a confusing numeric input box
        selected_label = st.selectbox(
            "Target Modification Region:", 
            options=list(position_options.keys())
        )
        
        # 3. Silently extract the true index vector for the RDKit engine backend
        atom_vector = position_options[selected_label]
        
        can_run = bool(st.session_state.rd_receptor and st.session_state.rd_ligand)
        if st.button("🚀 Execute 10-Pose Redesign Optimization Array", type="primary", disabled=not can_run):
            with st.spinner("Processing deep optimization forward layers..."):
                results_df = generate_dynamic_derivatives(st.session_state.rd_parent_smiles, atom_vector)
                st.session_state.rd_library = results_df
                st.rerun()
