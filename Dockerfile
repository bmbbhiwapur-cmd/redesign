# Dockerfile for InSilico BioSphere with full AutoDock Vina support
# Use this on Hugging Face Spaces (Docker SDK), Render, or Railway

FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install scientific stack via conda-forge (handles C++ deps cleanly)
RUN conda install -c conda-forge -y \
    python=3.10 \
    rdkit \
    openbabel \
    numpy \
    pandas \
    pillow \
    && conda clean -afy

# Install Python-only packages via pip
RUN pip install --no-cache-dir \
    streamlit \
    py3Dmol \
    stmol \
    meeko \
    vina

# Copy application
COPY app.py /app/app.py

# Hugging Face Spaces uses port 7860; Streamlit Cloud uses 8501
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
