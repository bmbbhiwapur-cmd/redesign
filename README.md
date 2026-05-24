# 🧬 InSilico BioSphere — AI Small-Molecule Redesign Studio

A Streamlit application for phytochemical scaffold redesign, fragment-based derivative generation, and AutoDock Vina molecular docking.

**Developed by:** Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India.

---

## ⚡ Quick Local Run

```bash
git clone <your-repo-url>
cd insilico_biosphere

# Option A — Conda (recommended, handles Vina + OpenBabel cleanly)
conda create -n biosphere python=3.10 -y
conda activate biosphere
conda install -c conda-forge rdkit openbabel -y
pip install streamlit py3Dmol stmol meeko vina pandas numpy pillow

streamlit run app.py
```

```bash
# Option B — pip only (no Vina/OpenBabel — demo mode only)
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Deployment Guide

Three deployment paths, ranked by how well they support your stack:

### 🥇 Best: Hugging Face Spaces (Docker SDK) — Full Vina Support

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Docker** as the SDK
3. Push these files to the Space repo:
   - `app.py`
   - `Dockerfile`
   - `README.md`
4. The build takes ~10 minutes the first time. Vina + OpenBabel + RDKit all work.

### 🥈 Render.com / Railway — Also Full Support

1. Connect your GitHub repo
2. Select **Docker** as build method (uses the included `Dockerfile`)
3. Deploy. First build ~10 minutes.

### 🥉 Streamlit Community Cloud — Demo Mode Only

Streamlit Cloud cannot reliably build the Vina wheel, so the app will run in heuristic demo mode.

1. Push to GitHub with these files:
   - `app.py`
   - `requirements.txt` (the **basic** one, without `vina`/`meeko`)
   - `packages.txt`
2. Deploy at https://share.streamlit.io
3. The app will load, but display "Vina unavailable — demo mode."

---

## 📂 File Overview

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Basic pip deps (for Streamlit Cloud) |
| `requirements-full.txt` | Full deps including Vina (for Docker) |
| `packages.txt` | apt packages for Streamlit Cloud |
| `Dockerfile` | Container build for HF Spaces / Render / Railway |

---

## 🧪 How to Use

1. **Load Protein:** Enter a 4-letter PDB code (e.g., `2AMB`) or upload a `.pdb` / `.pdbqt`
2. **Convert (if needed):** Click "Convert PDB to PDBQT & Load" — requires OpenBabel
3. **Load Ligand:** Enter SMILES (default: Paracetamol). Try `O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12` for Quercetin
4. **Pick Mechanism:** Covalent substitution or co-crystal formulation
5. **Pick Target Atom:** Select from valid sites
6. **Generate Derivatives:** Creates 4 fragment-substituted variants
7. **Dock:** Select a derivative and run Vina (real or demo)
8. **Inspect Pose:** Browse poses in the 3D viewer; contact residues are auto-detected

---

## ⚠️ Important Notes

- **Synthesis routes, yield predictions, and FTIR peaks** for the fragment library are pedagogical heuristics, not literature-validated values. They illustrate scaffold-class trends for teaching.
- **Docking energies** from Vina are real when Vina is installed; otherwise the app shows clearly-labeled heuristic estimates.
- **Contact residues** are now detected from actual pose coordinates (within 4 Å of the ligand), not hardcoded.

---

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'vina'"**
→ Vina isn't installed. App will run in demo mode. To get real docking, use Conda or Docker.

**"libopenbabel.so.7: cannot open shared object file"**
→ Missing system library. Add `libopenbabel-dev` to `packages.txt` (Streamlit Cloud) or use Docker.

**App loads but PDB conversion fails**
→ OpenBabel isn't installed. Upload pre-prepared `.pdbqt` files instead.

**Build times out on Streamlit Cloud**
→ Vina is too heavy. Remove `vina` and `meeko` from requirements; deploy via Docker instead.

---

## 📜 License

MIT — feel free to use for teaching and research. Attribution appreciated.
