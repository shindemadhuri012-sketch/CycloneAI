# Dataset Access Protocols & Storage Guidelines

This guide details access mechanisms, authentication requirements, download procedures, and storage allocations for CycloneAI.

---

## 1. NOAA NCEI IBTrACS v04r01 Access

### Access Classification: **Open Public Access (Zero Authentication)**

The North Indian Ocean best-track archive is hosted openly by NOAA NCEI. No API keys, tokens, or user accounts are required.

### Direct Download Endpoints:
- **North Indian Ocean Basin Subset (CSV)**:  
  `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv`  
  *Size*: ~5.2 MB (Contains all recorded storms in Bay of Bengal and Arabian Sea).
- **Global Archive (All Basins CSV)**:  
  `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv`  
  *Size*: ~250 MB uncompressed.

### Automated Python Command:
CycloneAI provides an automated retrieval script:
```powershell
python ml/datasets/download/download_ibtracs.py --basin NI
```
The file is automatically saved into `data/external/ibtracs.NI.list.v04r01.csv`.

---

## 2. TCIR Satellite Benchmark Access

### Access Classification: **Open Academic Research Access**

Hosted on university and academic research mirrors.

### Retrieval Procedure:
1. Primary Academic Mirror: [National Taiwan University TCIR Portal](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/)
2. Alternative Mirrors: Zenodo Open Science Repository & Kaggle Research Datasets (`tcir-dataset`).
3. Formats: Standard HDF5 containers (`TCIR-ALL.h5`, `TCIR-train.h5`, `TCIR-val.h5`, `TCIR-test.h5`).
4. Size:
   - Full global dataset: ~18 GB (70,000+ four-channel frames).
   - Sample development partition: ~500 MB (curated for local development).

---

## 3. ISRO MOSDAC INSAT-3D/3DR Access Protocol

### Access Classification: **Restricted — Free Registration Required**

ISRO data cannot be downloaded anonymously via open scripts. The Space Applications Centre mandates user identification.

### Registration Procedure:
1. Navigate to the official portal: [https://www.mosdac.gov.in](https://www.mosdac.gov.in).
2. Click **Register** on the top navigation bar. Fill in academic institution details (Student / Smart India Hackathon).
3. Confirm email verification to activate the account.
4. Log in and navigate to **Data Access → Satellite Data Order**.
5. Select:
   - **Satellite**: INSAT-3D or INSAT-3DR
   - **Sensor**: Imager
   - **Product**: L1B or L1C Standard (`3SIMG_L1B_STD`)
   - **Bands**: TIR-1 (10.8 µm), TIR-2 (12.0 µm), WV (6.8 µm)
   - **Date Range**: Select historical Indian cyclone dates (e.g., Cyclone Amphan: 16–21 May 2020; Cyclone Biparjoy: 06–16 June 2023).
6. Submit the order. An email notification with direct download links is delivered within 15–60 minutes.
7. Place downloaded HDF5 granules into `data/raw/insat3d/`.

---

## 4. Hardware & Storage Requirements for Development

| Resource Component | Minimal Development Mode | Standard Student Training Mode | Full Scale Research Mode |
| :--- | :--- | :--- | :--- |
| **Disk Storage** | < 100 MB | 2 GB – 5 GB | 30 GB – 50 GB |
| **RAM** | 8 GB | 16 GB | 32 GB+ |
| **Compute** | Laptop CPU | Google Colab Free T4 GPU / Kaggle | Cloud GPU (A100 / RTX 4090) |
| **Bandwidth** | ~10 MB download | ~1 GB download | ~25 GB download |

*CycloneAI defaults to the Minimal Development Mode and Standard Student Training Mode, ensuring feasibility on standard consumer laptops.*
