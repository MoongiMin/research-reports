# 🧠 Research Reports

### 🎧 Sound-Based Machine State Monitoring using AI
**Author:** Moon Gi Min  
**Advisor:** Prof. Jun, Martin Byung-Guk (Purdue ME)  
**Course:** ME49800ZU – Smart Manufacturing (Fall 2024)

#### 📄 Abstract
This study investigates sound-based monitoring for additive manufacturing machines using a custom sound sensor and a CNN model. Audio data from a Renishaw AM-400 was analyzed with features such as RMS, spectral centroid, and FFT energy.  
The CNN distinguishes Printing vs. Off states, showing that sound can be a reliable, low-cost method for smart manufacturing monitoring.

📘 [**Download Full Report (PDF)**](ME498_Research_Report%20git.pdf)


---

### 🔧 Key Highlights
- Built a **custom sound sensor** (modified stethoscope + Raspberry Pi)  
- Extracted 11 audio features (RMS, spectral centroid, FFT energy, etc.)  
- Implemented **CNN model with dropout & early stopping**  
- Achieved **100 % accuracy** on unseen test data  
- Proposed future work on **multi-state detection & real-time monitoring**
