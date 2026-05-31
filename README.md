# RNA 3D Structure Prediction: Hybrid TBM & Protenix Pipeline

An advanced, high-performance hybrid pipeline designed for the **Stanford RNA 3D Folding-2** Kaggle competition. This system combines **Template-Based Modeling (TBM)** with a deep learning inference engine (**Protenix**) to predict accurate 3D coordinates of RNA molecules at scale, efficiently handling long sequences through intelligent chunking and Kabsch alignment stitching.

## 🌟 Key Features

* **Hybrid Prediction Strategy:** Seamlessly blends fast Template-Based Modeling with deep learning inference, falling back to Protenix only when sequence similarity drops below the specified threshold.
* **Intelligent Sequence Chunking:** Overcomes hardware memory limitations by automatically splitting long RNA sequences into overlapping chunks and dynamic multi-pass inference.
* **Advanced Coordinate Stitching:** Implements the **Kabsch Algorithm** to align and structurally stitch predicted chunks back into a globally consistent 3D structure using weighted linear blending.
* **Domain-Specific Refinement:** Includes an adaptive physics-informed constraint mechanism to post-process predictions, enforcing realistic RNA bio-physical bond distances and spatial distributions.
* **Data Augmentation Ensembling:** Utilizes customized structural perturbations (Hinge rotation, chain jittering, and smooth wiggle animations) to generate robust ensemble submissions.

---

## 🏗️ Pipeline Architecture

1. **Phase 1 (TBM):** Searches the template pool using a global pairwise aligner. If highly similar sequences are found, coordinates are adapted and augmented to fill the required sample slots.
2. **Phase 2 (Protenix Inference):** Sequences lacking sufficient templates are queued, formatted into JSON, and fed into an optimized Protenix-v1 model on dual GPUs.
3. **Phase 3 (Stitching & Combination):** Chunked outputs are stitched via Kabsch alignment, refined using dynamic constraints, combined with TBM tracks, and exported.

---

## 💻 Environment Setup & Dependencies

The pipeline is fully optimized for Kaggle environments running PyTorch with CUDA acceleration. It loads essential pre-compiled wheels offline to ensure quick setup:

* `biopython`
* `biotite`
* `rdkit`
* `ml_collections`

### Local Execution
To run the inference script natively, ensure you point the environment variables to your data directory:
```bash
export TEST_CSV="/path/to/test_sequences.csv"
export PROTENIX_CODE_DIR="/path/to/protenix-v1"
python main.py
