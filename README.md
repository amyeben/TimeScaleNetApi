# TimeScaleNet API & App 

**Empowering Environmental Sound Recognition for Hearing Accessibility**

## Abstract  
TimeScaleNet leverages **Deep Learning** techniques to process raw audio waveforms with multi-resolution analysis. Initially introduced by researchers at the **CNAM**, it features **BiquadNet** (a learnable passband IIR filter layer) and **FrameNet** (a residual network with depthwise separable convolutions) to analyze sounds at sample and frame scales. Our project optimizes TimeScaleNet for real-world deployment on portable devices, achieving significant accuracy improvements.

---

## 🏗️ **Architecture**

### Project Structure:
```
├── app/             # Next.js frontend application
├── backend/         # FastAPI backend for prediction and preprocessing
├── public/          # Static files for the frontend
├── timescalenet_model.h5  # Optimized ESC-10 TimeScaleNet model
├── timescalenet_urbansound8k.h5  # Optimized UrbanSound8K model
```

### Technologies Used:
- **Frontend**: [Next.js](https://nextjs.org/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Deep Learning Framework**: [TensorFlow](https://www.tensorflow.org/)
- **Audio Preprocessing**: [Librosa](https://librosa.org/)

---

## 🚀 **How to Run the Project**

### Prerequisites:
- Python 3.8+
- Node.js 14+
- TensorFlow and other dependencies (`pip install -r requirements.txt`)

### Steps to Deploy:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/TimeScaleNetApi.git
   cd TimeScaleNetApi
   ```

2. Start the FastAPI backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

3. Start the Next.js frontend:
   ```bash
   cd ../app
   npm install
   npm run dev
   ```

4. Access the application at:
   - Frontend: [http://localhost:3000](http://localhost:3000)  
   - Backend API: [http://localhost:8000](http://localhost:8000/docs)  

---

## 📊 **Key Achievements**

### Datasets:
1. **ESC-10**: Environmental Sound Classification (10 classes).  
   - Accuracy: **89%**  
   - F1 Score: **0.88**

2. **UrbanSound8K**: Urban sound recognition (10 classes).  
   - Accuracy: **94%**  

### Improvements:
- **ESC-10**: Boosted from 69% to 89% with enhanced contextual learning.  
- **UrbanSound8K**: Achieved state-of-the-art performance with 94% accuracy.

---

## 🤝 **Acknowledgments**
- **Researchers at CNAM** for developing TimeScaleNet and laying the groundwork for our project.
- Our **coach and mentors** for their continuous support and guidance.
- **Our team**, whose collaboration and dedication drove this project to success.


© Timescalenet : A Multiresolution Approach for Raw Audio Recognition : https://ieeexplore.ieee.org/document/8682378 
