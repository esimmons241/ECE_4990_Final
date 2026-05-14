PCB Defect Detection using Deep Learning via U Net

**Automated Quality Control System for Printed Circuit Board Manufacturing**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![F1-Score](https://img.shields.io/badge/F1--Score-90.2%25-success.svg)]()

---

## 🎯 Project Overview

An end-to-end deep learning system for **automated PCB defect detection** that combines computer vision with domain expertise. This project demonstrates the practical application of AI in industrial quality control, achieving **90.2% F1-score** on multi-label defect classification.

**Built as a portfolio project showcasing:**
- 🧠 Deep Learning & Computer Vision expertise
- ⚙️ Industrial ML system design
- 🔧 Electrical Engineering domain knowledge
- 📊 Data-driven problem solving

![Sample Detection](results/detection_pipeline_4stage.png)

---

## ⚡ Key Highlights

- ✅ **90.2% F1-Score** on DeepPCB benchmark dataset
- 🎯 **6 Defect Types:** Open circuits, shorts, mousebites, spurs, spurious copper, pin-holes
- ⚙️ **Real-time inference:** 20ms per image (50 images/second)
- 📈 **Stable training:** Dropout + Weight Decay regularization
- 🔄 **Production-ready:** Automated QA report generation
- 💰 **ROI:** 6-12 months payback period

---

## 🏗️ System Architecture
```
Input PCB Image (640×640px)
         ↓
    Preprocessing
         ↓
  ResNet-18 CNN Backbone
   (Transfer Learning)
         ↓
   Dropout Layer (0.5)
         ↓
  Multi-Label Classification
    (6 defect classes)
         ↓
   Sigmoid Activation
         ↓
  Confidence Scores (0-1)
         ↓
  Threshold Decision (0.5)
         ↓
Quality Control Report
```

---

## 📊 Performance Metrics

### Overall Performance

| Metric | Score | Industry Target |
|--------|-------|-----------------|
| **F1 Score** | **90.2%** | 85-95% ✅ |
| **Precision** | 83.4% | >80% ✅ |
| **Recall** | 99.8% | >95% ✅ |
| **Inference Time** | 20ms/image | <100ms ✅ |

![Training Curves](results/complete_training_comparison.png)

### Per-Class Performance

| Defect Type | F1 Score | Notes |
|-------------|----------|-------|
| **Open Circuit** | 98.5% | Excellent detection ⭐ |
| **Short Circuit** | 87.1% | Good performance |
| **Mousebite** | 90.5% | Strong recall |
| **Spur** | 85.5% | Challenging class |
| **Spurious Copper** | 89.3% | Very good |
| **Pin-hole** | 93.4% | Excellent precision |

![Confusion Matrices](results/confusion_matrices_regularized.png)

---



## 🎨 Visualizations

### Detection Pipeline
![4-Stage Pipeline](results/detection_pipeline_4stage.png)
*Complete detection pipeline showing template comparison to AI classification*

### Results Summary
![Results Grid](results/detection_results_grid.png)
*Quick summary of detection results across multiple samples*

### Model Predictions
![Side-by-Side Predictions](results/model_predictions_sidebyside.png)
*Detailed model predictions with confidence scores*

### Performance Analysis
![Training Curves](results/complete_training_comparison.png)
*Training stability comparison between models*

![Confusion Matrices](results/confusion_matrices_regularized.png)
*Per-class confusion matrices showing detection accuracy*

![Per-Class Performance](results/per_class_performance.png)
*Detailed precision-recall analysis by defect type*

---

## 🧪 What Makes This Project Unique

### 1. **Regularization Strategy** 🎓

Implemented **Dropout (0.5) + L2 Weight Decay (1e-4)** to prevent overfitting:

**Result:** Training stability improved by 51%

### 2. **Template-Based Quality Reports** 📋

Unlike generic AI models (BLIP), we use **domain-specific templates**:
```
PCB QUALITY INSPECTION REPORT
═══════════════════════════════════════
Status: ⚠️ FAIL - HIGH Severity

DETECTED DEFECTS: 2
  • SHORT CIRCUIT (Confidence: 94%)
    → Unintended electrical connection
    → Risk: Component damage, fire hazard

  • OPEN CIRCUIT (Confidence: 87%)
    → Discontinuity in electrical path
    → Risk: Non-functional board

RECOMMENDATIONS:
  1. URGENT: Do NOT proceed to assembly
  2. Review etching process parameters
  3. Inspect batch for similar defects
```

**Why This Matters:** 100% technical accuracy vs. <10% with generic BLIP model

### 3. **Precision-Recall Optimization** ⚖️

Deliberately prioritized **high recall (99.8%) over precision (83.4%)** because:

| Error Type | Business Impact |
|-----------|----------------|
| **False Negative** (missed defect) | Board ships to customer → Field failure → $1,000+ cost |
| **False Positive** (false alarm) | Extra 2-min inspection → $2 cost |

**Decision:** Better to have false alarms than miss critical defects!

---

## 🛠️ Technical Implementation

### Tech Stack

**Core Framework:**
- Python 3.9+
- PyTorch 2.0+
- torchvision (ResNet-18)

**Data Processing:**
- OpenCV - Image preprocessing
- NumPy - Numerical operations
- Pandas - Data manipulation

**Visualization:**
- Matplotlib & Seaborn
- Confusion matrices
- Training curves

**Dataset:**
- DeepPCB (1,500 PCB image pairs)
- 6 defect classes
- 640×640px resolution

### Model Architecture

ResNet-18 with regularized classifier head:
- Pretrained on ImageNet for transfer learning
- Dropout layer (0.5) to prevent overfitting
- Multi-label output for simultaneous defect detection
- Sigmoid activation for independent class probabilities

### Training Configuration

- **Epochs:** 20 (with early stopping)
- **Batch size:** 16
- **Learning rate:** 0.001
- **Optimizer:** Adam with weight decay (1e-4)
- **Loss:** Binary Cross-Entropy
- **Scheduler:** ReduceLROnPlateau

**Data Augmentation:**
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast)

---

## 📁 Project Structure
```
pcb-defect-detection/
├── notebooks/
│   ├── 01_EDA_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_threshold_optimization.ipynb
│   └── 04_genai_integration.ipynb
├── src/
│   └── dataset_utils.py
├── models/
│   ├── best_model.pth
│   └── best_model_regularized.pth
├── results/
│   ├── complete_training_comparison.png
│   ├── confusion_matrices_regularized.png
│   ├── per_class_performance.png
│   ├── sample_predictions.png
│   └── IEEE_Project_Report.md
├── DeepPCB/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- 4GB+ RAM
- GPU recommended (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/pcb-defect-detection.git
cd pcb-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download DeepPCB dataset
git clone https://github.com/tangsanli5201/DeepPCB.git
```

### Quick Start

**Use Pretrained Model:**
```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = PCBDefectClassifier()
model.load_state_dict(torch.load('models/best_model_regularized.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('path/to/pcb_image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = model(img_tensor)

# Results
defects = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
for defect, conf in zip(defects, predictions[0]):
    if conf > 0.5:
        print(f"{defect}: {conf:.2%} confidence")
```

---

## 💡 Key Learnings & Insights

### 1. **Generic AI ≠ Specialized Domains**

**Experiment:** Tried BLIP (vision-language model) for automated defect descriptions

**Result:** Complete failure ❌
```
Input:  PCB with open circuit defect
BLIP:   "a circuit board with chip chip chip chip..."
Needed: "Open circuit detected in trace at confidence 94%"
```

**Why it failed:**
- Trained on natural images (cats, dogs), not technical PCBs
- No PCB-specific vocabulary
- Can't recognize circuit patterns

**Lesson:** For technical domains, **domain knowledge + simple rules > fancy AI without fine-tuning**

### 2. **Training Stability Matters**

Original training showed erratic validation loss with dramatic dips. Solutions tried:

| Approach | Impact |
|----------|--------|
| More epochs (20) | ✅ Better convergence |
| Dropout + Weight Decay | ✅✅ **Significant improvement** |

**Result:** Validation loss std dev reduced by 51%

### 3. **Default Threshold Often Optimal**

Explored 1000+ threshold combinations. Finding: **Default 0.5 was already near-optimal!**

This indicates:
- Model calibration is good
- Transfer learning worked well
- No need for complex threshold tuning

### 4. **Business Context > Raw Accuracy**

**Question:** Should we optimize for 92% F1 vs current 90.2%?

**Analysis:**
- Development time: 8+ hours
- Performance gain: ~2% F1
- Business impact: 11% fewer manual reviews
- Annual savings: ~$300

**Decision:** **Not worth it.** Time better spent on deployment prep.

---

## 🎯 Business Impact & ROI

### Cost-Benefit Analysis

| Aspect | Manual Inspection | Our AI System | Savings |
|--------|------------------|---------------|---------|
| **Setup Cost** | $0 | $3,000-5,000 | - |
| **Annual Labor** | $120K-300K | $10K maintenance | **$110K-290K/year** |
| **Throughput** | 20-30 boards/hour | 2000+ boards/hour | **50-100× faster** |
| **Detection Rate** | ~85% | **99.8%** | Fewer field failures |
| **Consistency** | Varies | 24/7 consistent | No degradation |

**ROI Timeline:** 6-12 months

### Target Industries

- ✅ Consumer electronics (smartphones, IoT devices)
- ✅ Automotive (ADAS, EV battery management)
- ✅ Medical devices (pacemakers, imaging equipment)
- ✅ Aerospace & defense (avionics, satellites)
- ✅ Contract manufacturers (high-volume production)

---

## 🔮 Future Enhancements

### Short-Term (1-3 months)

- [ ] **Web Dashboard** - Flask/Streamlit UI for inspectors
- [ ] **Defect Localization** - Add bounding boxes (YOLO v8)
- [ ] **Confidence Calibration** - Platt scaling for better probabilities

### Medium-Term (3-6 months)

- [ ] **Active Learning** - Continuously improve with production data
- [ ] **Explainable AI** - GradCAM visualization showing detection reasons
- [ ] **Ensemble Models** - Combine multiple architectures

### Long-Term (6-12 months)

- [ ] **Root Cause Analysis** - ML model to predict defect causes
- [ ] **End-to-End Platform** - Integrate with MES/ERP systems
- [ ] **Edge Deployment** - On-device inference with TensorRT/ONNX

---

## 🤔 Limitations & Considerations

**Honest Assessment:**

### Dataset Limitations
- **Size:** 1,500 images vs 100K+ in commercial systems
- **Diversity:** Single PCB type; may not generalize to flex PCBs, HDI, RF boards
- **Class Imbalance:** Real manufacturing has 100:1 defect-to-clean ratios

### Architecture Limitations
- **No Localization:** Detects presence, not exact defect location
- **Fixed Input Size:** 640×640px; may miss small defects on large boards

### Deployment Considerations
- **False Alarms:** Some false positives (acceptable in QC)
- **Novel Defects:** Model only knows 6 trained classes
- **Environmental Factors:** Lighting, camera angle affect performance

---

## 📚 References & Resources

### Academic Papers
1. Tang et al., "PCB Defects Detection Using Deep Learning", arXiv 2019
2. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

### Datasets
- **DeepPCB:** [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)

### Tools & Frameworks
- **PyTorch:** [pytorch.org](https://pytorch.org)
- **OpenCV:** Image processing

---

## 🙏 Acknowledgments

- **DeepPCB Team** - For open-sourcing the dataset
- **PyTorch Community** - Excellent framework
- **ResNet Authors** - Transfer learning foundation

---

## 📧 Contact

**[Your Name]**
🎓 Electrical & Electronics Engineering Graduate
💼 Aspiring ML Engineer specializing in Industrial AI

📧 Email: your.email@example.com
🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)
📂 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ⭐ If You Found This Useful

Give it a star ⭐ on GitHub!

---

<div align="center">

**Built with 🔥 by an EEE grad exploring the intersection of hardware and AI**

*"The best AI solution balances accuracy, speed, cost, and interpretability."*

</div>
