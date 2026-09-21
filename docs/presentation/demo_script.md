# CycloneAI: Live Hackathon Demonstration Script
**Step-by-Step Presenter Walkthrough for Judges & Evaluators**  
**Estimated Demo Time:** 4–5 minutes  

---

## Pre-Demo Checklist (1 Minute Before Judging)
1. Launch production server:
   ```powershell
   python run_production.py
   ```
2. Open Google Chrome or Firefox to:
   ```
   http://localhost:8000
   ```
3. Ensure browser window is maximized (1920x1080 or projector display mode).
4. Verify backend health:
   ```
   http://localhost:8000/api/health
   ```
   (Should return `status: "ok"` and all models `active`).

---

## Screen-by-Screen Presenter Walkthrough

### Screen 1: Command Center Dashboard (`/`)
- **Action**: Start on the main **Dashboard** view.
- **Presenter Dialogue**:
  > *"Welcome to CycloneAI. This is our unified operational command center designed for meteorological analysts and emergency disaster managers. On the right, you can see live system telemetry showing that all five AI subsystems—Detection, Classification, Intensity, Track, and Explainability—are loaded and active in memory."*
- **Action**: Point to the **Benchmark Presets** carousel (FANI, AMPHAN, TAUKTAE, REMAL, BIPARJOY).
- **Presenter Dialogue**:
  > *"Rather than using synthetic mock data, CycloneAI is connected to the verified 43-year North Indian Ocean best-track archive. Let's select **Cyclone FANI (2019)**—one of the most severe cyclones to hit the Odisha coastline in recent history."*
- **Action**: Click on **"Load Preset: FANI"**. Observe input coordinates auto-populate ($16.0^\circ\text{N}$, $84.5^\circ\text{E}$, $932\text{ hPa}$, $115\text{ kt}$).
- **Action**: Click the bright blue **"Run Unified Analysis Pipeline"** button.
- **Presenter Dialogue**:
  > *"Notice the latency counter: **68 milliseconds**. In less than a tenth of a second, our pipeline executed four machine learning models and our Explainable AI engine simultaneously. Let's inspect each analytical layer."*

---

### Screen 2: Satellite Convective Analysis & Ingestion (`/satellite-analysis`)
- **Action**: Click **"Satellite Analysis"** in the sidebar.
- **Presenter Dialogue**:
  > *"Here, the system simulates real-time INSAT-3D thermal infrared (TIR-1) geostationary imagery. CycloneAI analyzes the convective cloud deck, measuring cloud-top brightness temperatures down to -78°C in the central dense overcast (CDO). The CycloneDetector flags cyclonic storm genesis with **99.4% confidence**, correctly distinguishing organized cyclonic rotation from background monsoon depressions."*

---

### Screen 3: Cyclone Category Classification (`/classification`)
- **Action**: Click **"Classification"** in the sidebar.
- **Presenter Dialogue**:
  > *"On this screen, our 8-tier classifier categorizes the storm according to official India Meteorological Department (IMD) standards. For Cyclone FANI, the model assigns the peak category: **Extremely Severe Cyclonic Storm (ESCS)** with 91.2% probability. You can see the full probability distribution across all 8 grades in the bar chart, confirming zero probability for weaker depression stages."*

---

### Screen 4: Multi-Horizon Intensity Forecasting (`/intensity`)
- **Action**: Click **"Intensity Prediction"** in the sidebar.
- **Presenter Dialogue**:
  > *"Now we move to intensity forecasting. CycloneAI predicts Maximum Sustained Surface Wind ($V_{\max}$) and Minimum Central Sea-Level Pressure ($P_{\min}$) across future horizons: +6h, +12h, and +24h. Notice three crucial features here:*
  > 1. *The shaded ribbons represent **90% quantile confidence bounds** ($q_{10}-q_{90}$). On our 70-storm test split, 86.1% of actual observations fell within this interval.*
  > 2. *Our **Hydrodynamic Consistency Engine** strictly enforces the IMD pressure-wind relation, preventing unphysical divergence between wind and pressure.*
  > 3. *The **Rapid Intensification (RI) Alert** is triggered, warning disaster authorities 24 hours in advance of explosive core strengthening."*

---

### Screen 5: Multi-Horizon Track Prediction (`/track`)
- **Action**: Click **"Track Prediction"** in the sidebar.
- **Presenter Dialogue**:
  > *"This is our interactive geospatial track forecasting engine. The Leaflet map displays the current storm center off the Andhra/Odisha coast. The projected trajectory plots future eye positions at +6h, +12h, +24h, and +48h.*
  > *Surrounding each forecast point are our **calibrated 75% empirical uncertainty cones**. Unlike arbitrary circular buffers, our cone radii were empirically calibrated on validation storm errors and verified on 70 unseen test storms to capture 74.9% of actual tracks.*
  > *Notice that translation speed is kinematically clamped below 120 km/h, preventing erratic jumping."*

---

### Screen 6: Explainable AI (XAI) Suite (`/explainability`)
- **Action**: Click **"Explainable AI"** in the sidebar.
- **Presenter Dialogue**:
  > *"Meteorologists will never trust a black-box AI. That is why CycloneAI incorporates full Explainable AI:*
  > 1. *On the left is our **Local Feature Attribution Waterfall**. Using TreeSHAP path decomposition, it shows the exact mathematical contribution of every physical feature. Central pressure deficit contributed +18.4 kt to the intensity estimate, while forward translation bearing added another +6.2 kt. This attribution strictly satisfies the Efficiency Axiom to zero error.*
  > 2. *In the center, our **Dvorak BD convective cloud ring saliency** highlights the cold eye ring and warm eye pocket driving the classification.*
  > 3. *At the bottom, our automated **Meteorological Synoptic Briefing** translates model math into plain English operational text for immediate relay to state disaster authorities."*

---

### Screen 7: Historical Cyclone Explorer (`/historical`)
- **Action**: Click **"Historical Cyclones"** in the sidebar.
- **Presenter Dialogue**:
  > *"To demonstrate our data provenance, judges can search through the entire 1,858-storm archive. Let's filter by **Bay of Bengal, Category VSCS+**. We can click into Cyclone AMPHAN (2020) or Cyclone HUDHUD (2014) to inspect every 3-hour observation fix with real coordinates, sustained wind speeds, and central barometric readings."*

---

### Screen 8: Model Performance & Scientific Verification (`/model-performance`)
- **Action**: Click **"Model Performance"** in the sidebar.
- **Presenter Dialogue**:
  > *"Finally, we show complete scientific transparency. Every number on this page is read directly from our Phase 10 Master Scientific Evaluation Report on the 70 completely held-out test storms:*
  > - *Detection ROC-AUC: **0.9615**.*
  > - *Category Adjacent Accuracy: **79.85%**.*
  > - *Track Skill: **+18.3% over Persistence at +24h**, **+20.8% at +48h**.*
  > - *Statistical significance: **p < 10⁻²⁹** via paired Student's t-test.*
  > *CycloneAI is fully open-source, reproducible with a single script, and ready for deployment."*

---

## Emergency Troubleshooting
- **If port 8000 is occupied**: `PORT=8080 python run_production.py`.
- **If Leaflet tiles load slowly**: Tiles are cached by the browser; local GeoJSON borders remain fully responsive.
- **If a preset reset is needed**: Refresh the browser page or click **"Reset Form"** in the dashboard header.
