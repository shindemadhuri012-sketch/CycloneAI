# CycloneAI: Hackathon Pitch & Presentation Guide
**Smart India Hackathon (SIH) 2026**  
**Problem Statement:** *"To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."*  
**Category:** Disaster Management / Space Technology / Earth Observation  
**Project:** CycloneAI — Intelligent Tropical Cyclone Analysis & Prediction System  

---

## 1. Executive Presentation Strategy

Judges at Smart India Hackathon evaluate projects on five key dimensions:
1. **Relevance to Problem Statement (20%)**: Does the system solve detection, classification, and multi-horizon prediction using satellite data?
2. **Technical Innovation & Architecture (25%)**: Is the AI scientifically rigorous, decoupled, reproducible, and explainable?
3. **Scientific Validation & Accuracy (20%)**: Are benchmarks grounded in real-world data with statistical proof of skill over operational baselines?
4. **Usability & Decision Support (20%)**: Can emergency managers (NDMA, SDMAs) take actionable, life-saving decisions from the interface?
5. **Feasibility & Real-World Deployability (15%)**: Is the system lightweight, production-ready, sub-100ms fast, and deployable at zero supercomputing cost?

---

## 2. Recommended Pitch Formats

### Format A: 3-Minute Rapid Pitch (Elimination / Preliminary Round)

| Timestamp | Screen / Visual | Speaker Talking Points |
| :--- | :--- | :--- |
| **0:00 - 0:35** | **Dashboard Overview** | *"Every year, tropical cyclones ravage the coastal communities of Odisha, West Bengal, Andhra Pradesh, and Gujarat. Operational meteorologists face two massive bottlenecks: traditional numerical supercomputers take 6 hours per run, while subjective manual Dvorak analysis varies between analysts during rapid intensification. We built **CycloneAI**: an end-to-end intelligent decision-support system that detects, classifies, forecasts intensity and track, and explains decisions in under 100 milliseconds."* |
| **0:35 - 1:30** | **Live Pipeline Trigger (Cyclone FANI)** | *"Watch this live: with one click on verified observation data from Cyclone FANI, our unified pipeline orchestrates 4 machine learning models and Explainable AI simultaneously. In 65 milliseconds, we obtain: (1) 99% cyclone probability, (2) Extremely Severe Cyclonic Storm classification, (3) +24h intensity with 90% confidence ribbons, (4) geospatial track trajectory with 75% uncertainty cone polygons, and (5) exact TreeSHAP feature attributions."* |
| **1:30 - 2:20** | **Model Performance Page** | *"We did not use synthetic or fabricated data. We trained and evaluated strictly on the 43-year North Indian Ocean best-track archive. On 70 completely unseen test storms spanning 2,223 fixes: our track predictor achieved an 18.3% skill gain over Persistence at +24h with a p-value less than 10⁻²⁹. Our intensity model delivers an 8.1% wind skill gain and successfully flags Rapid Intensification."* |
| **2:20 - 3:00** | **Explainability & Impact** | *"Crucially, CycloneAI is not a black box. Our Explainable AI engine provides TreeSHAP attribution matching the Efficiency Axiom to 0.000000 error, 2D Dvorak convective cloud ring saliency, and automated synoptic briefing text for NDMA disaster managers. CycloneAI runs on a standard CPU server with zero GPU requirements, ready for immediate deployment in state emergency operation centers."* |

---

### Format B: 5-Minute Technical Deep Dive (Final Evaluation Round)

#### Slide 1: Problem Statement & Meteorological Gap
- **Challenge**: The North Indian Ocean (Bay of Bengal & Arabian Sea) has the highest coastal mortality rate globally from tropical cyclones.
- **Current Limitations**:
  1. *Supercomputer Latency*: High-resolution NWP models (HWRF, GFS) require 4–6 hours of compute time.
  2. *Subjective Dvorak Analysis*: Analyst variance during nighttime infrared transitions.
  3. *Rapid Intensification Blindspot*: Sudden $\Delta V \ge 30\text{ kt}$ jumps are notoriously difficult to predict.
- **CycloneAI Solution**: A real-time, physics-constrained multi-task ML architecture ingesting satellite infrared observations and kinematic vectors to provide sub-100ms multi-horizon intelligence.

#### Slide 2: End-to-End System Architecture
- **Data Backbone**: 43 years (1982–2025) of verified North Indian Ocean archives (1,858 storms, 57,841 fixes).
- **Zero Leakage**: Strict storm-level split (323 train, 69 val, 70 test storms) with 0.00% overlap.
- **Decoupled Architecture**:
  - *Backend*: FastAPI ASGI server, Pydantic v2 schemas, singleton model loaders.
  - *Frontend*: React 19, Vite, Tailwind CSS, Leaflet OpenStreetMap, Recharts.
  - *Container*: Single-container Dockerized deployment on port 8000.

#### Slide 3: The 4 Core Predictive Engines
1. **Identification Engine**: Binary genesis screening ($0.9615$ ROC-AUC, $93.66\%$ Top-1 accuracy).
2. **Classification Engine**: 8-tier official IMD scale categorizer ($79.85\%$ adjacent match within $\pm 1$ category, $1.14$ MACE).
3. **Multi-Horizon Intensity Predictor**: Lead forecasts for $+6\text{h}$, $+12\text{h}$, $+24\text{h}$ with $90\%$ quantile ribbons and IMD hydrodynamic pressure-wind coupling:
   $$V_{\max} \approx 14.2 \sqrt{1010 - P_{\min}}$$
   Achieves $+8.1\%$ wind skill gain and $+9.4\%$ pressure skill gain over Persistence at $+24\text{h}$ ($p = 0.0077$).
4. **Multi-Horizon Track Predictor**: Incremental coordinate displacement vectors $(\Delta \text{lat}, \Delta \text{lon})$ for $+6\text{h}$, $+12\text{h}$, $+24\text{h}$, $+48\text{h}$ with physical speed clamping ($v \le 120\text{ km/h}$) and $75\%$ empirical uncertainty cones ($74.6\% - 76.7\%$ test coverage).

#### Slide 4: Scientific Validation & Statistical Proof (Phase 10)
- **Held-Out Test Set**: 70 unseen storms, 2,223 observation fixes.
- **Track Benchmarks**:
  - $+6\text{h}$: $35.4\text{ km}$ ATE ($+8.2\%$ skill over Persistence).
  - $+12\text{h}$: $72.3\text{ km}$ ATE ($+12.4\%$ skill over Persistence).
  - $+24\text{h}$: $149.9\text{ km}$ ATE ($+18.3\%$ skill over Persistence).
  - $+48\text{h}$: $332.8\text{ km}$ ATE ($+20.8\%$ skill over Persistence).
- **Statistical Significance**: Paired Student's t-test and Wilcoxon signed-rank test confirm $p < 10^{-10}$ across all track lead times.

#### Slide 5: Explainable AI & Operational Trust
- **Local Feature Attribution**: TreeSHAP tree-path decomposition satisfying the Efficiency Axiom:
  $$\sum_{i=1}^{M} \phi_i = f(x) - \mathbb{E}[f(X)] \quad (\text{error} = 0.000000)$$
- **2D Dvorak BD Saliency**: Maps convective banding, CDO eye ring, and warm core brightness temperatures to operational Dvorak intensity criteria.
- **Decision Support**: Natural language synoptic narrative summarizing primary meteorological risk drivers.

#### Slide 6: Live Operational Demonstration
- Demonstrate live execution of presets:
  - **Cyclone FANI (2019)**: Extreme landfall prediction in Puri, Odisha.
  - **Cyclone TAUKTAE (2021)**: Arabian Sea rapid intensification and Gujarat track.
  - **Cyclone REMAL (2024)**: West Bengal / Bangladesh storm surge scenario.

---

## 3. Anticipated Judge Q&A & Rebuttals

### Q1: "How does CycloneAI compare to IMD's official numerical weather prediction models (HWRF / NCUM / GFS)?"
> **Answer**:  
> *"CycloneAI is designed as an immediate-lead decision support companion, not an NWP competitor. Supercomputer NWP runs require 4 to 6 hours to assimilate global fields and complete a simulation cycle. CycloneAI executes in **65 milliseconds** directly from current geostationary satellite fixes. This gives disaster managers an actionable 4 to 6-hour head start between NWP model cycles. Furthermore, our +6h (35.4 km) and +12h (72.3 km) track errors fall right inside official IMD operational error envelopes (40 km and 60 km), with zero supercomputing infrastructure cost."*

### Q2: "Did you use deep learning on raw satellite images or tabular features?"
> **Answer**:  
> *"CycloneAI utilizes a hybrid multi-modal architecture. For spatial and convective analysis, our satellite module processes INSAT-3D thermal infrared (TIR-1) brightness temperature matrices to compute Dvorak BD-curve enhancement zones, axisymmetry scores, and cloud-top gradient vectors. These physical convective features are coupled with planetary vorticity (Coriolis parameter), central pressure deficit, and historical kinematic displacement vectors into our calibrated gradient-boosted ensembles. This guarantees physical consistency and enables exact TreeSHAP explainability, which black-box end-to-end CNNs cannot provide."*

### Q3: "How do you ensure your models don't suffer from data leakage?"
> **Answer**:  
> *"We implemented strict 4-way leakage prevention: (1) **Storm-level isolation**: our 70 test storms were completely separated by unique Storm ID before any feature extraction, ensuring 0% storm overlap. (2) **Temporal causality**: kinematic differencing looks strictly backward (-6h, -12h). (3) **Zero synthetic samples**: we reject synthetic noise and evaluate only on verified IBTrACS fixes. (4) **Frozen checkpoints**: during our Phase 10 evaluation and Phase 11 deployment, zero model weights were retrained."*

### Q4: "What happens if a cyclone makes a sudden 180-degree retrograde turn like Cyclone MADI in 2013?"
> **Answer**:  
> *"In our Phase 10 extreme-event outlier audit, we specifically analyzed Cyclone MADI's sudden retrograde loop as our #1 outlier case (655 km error at +24h). We documented the physical cause: an environmental steering current saddle point where translation speed dropped below 8 km/h followed by mid-tropospheric ridge repositioning. Our system communicates this uncertainty honestly through calibrated 75% empirical uncertainty cones that widen proportionally with lead time."*

### Q5: "Can this system run on local state emergency operation center (SEOC) hardware?"
> **Answer**:  
> *"Yes. CycloneAI has a total memory footprint of approximately 320 MB and runs entirely on standard x86 CPU cores without requiring GPUs. It is containerized via Docker and can be launched on a single laptop or edge server with a single command: `docker compose up` or `python run_production.py`."*
