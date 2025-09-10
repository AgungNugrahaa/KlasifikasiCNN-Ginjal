## Detection and Classification of Kidney Diseases Using CT-Scanned Images  

## Project Description  
Proyek ini bertujuan untuk mendeteksi dan mengklasifikasikan penyakit ginjal (**Cyst, Normal, Stone, Tumor**) dari citra CT Scan.  
Dengan memanfaatkan teknik **deep learning** berbasis Convolutional Neural Network (CNN), aplikasi ini membantu mengotomatisasi proses klasifikasi untuk mendukung tenaga medis dalam diagnosis yang lebih cepat dan akurat.  

---

## Dataset  
- **Dataset Name**: [CT-KIDNEY-DATASET (Kaggle)](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/code)  
- **Categories**:  
  - Cyst  
  - Normal  
  - Stone  
  - Tumor  
- **Structure**: Dataset disusun dalam subdirektori per kelas, masing-masing berisi citra CT sesuai label.  

---

## ⚙️ Requirements  

### Python Libraries
- streamlit  
- tensorflow  
- numpy
- Pandas
- Seaborn
- matplotlib
- pillow
- os dan shutil 

Install semua dependensi dengan:  

```bash
pip install -r requirements.txt
```
```bash
project_root/
│
├── content/
│   └── save_model/           
│
├── notebooks/
│   └── notebook.ipynb        
│
├── app.py                    
├── requirements.txt          
└── README.md                 
```
## Workflow  

### 1. Data Preprocessing  
- Resize citra ke ukuran konsisten (244x244)  
- Normalisasi pixel → [0,1]  
- Penanganan class imbalance dengan **class weighting (inverse sqrt frequency)**  

### 2. Model Training  
- CNN custom (Conv2D → MaxPooling → Dense → Softmax)  
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  

### 3. Evaluation  
- Confusion Matrix  
- ROC Curve & Precision-Recall Curve  
- Accuracy, Precision, Recall, F1-Score  

### 4. Deployment  
- Export model ke **TensorFlow SavedModel**  
- Integrasi ke aplikasi **Streamlit**  

## How to Run  
### 1. Clone repository:  
```bash
git clone https://github.com/AgungNugrahaa/KlasifikasiCNN-Ginjal.git
```
### 2. Navigasi Project:
```bash
cd KlasifikasiCNN-Ginjal
```
### 3. Install Requirements:
```bash
pip install -r requirements.txt
```
### 4. Run Streamlit:
```bash
streamlit run app.py
```



