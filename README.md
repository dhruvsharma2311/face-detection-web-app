# 👁‍🗨 Face Recognition Attendance System

A **smart web-based attendance system** that uses **face recognition** to identify individuals from group photos and mark attendance automatically.  
Built using **Python**, **Streamlit**, **InsightFace**, and **OpenCV**.

---

## 🚀 Features

- 📸 Upload **group photos** to detect and recognize faces  
- 🧠 **InsightFace** for accurate face detection & embeddings  
- 🧮 **Cosine similarity** for face matching from a database  
- 🗂 Automatically tracks **present**, **absent**, and **unknown** individuals  
- 🧾 Generates **downloadable attendance summaries** in CSV format  

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, scikit-learn  
- **Face Recognition**: InsightFace, OpenCV  
- **Utilities**: NumPy, Pandas  

---

## 📁 Folder Structure

```
├── app.py                  # Main Streamlit application
├── main.py                 # Additional helper/processing script
├── working_model.ipynb     # Development & testing notebook
├── dataset/                # Sample dataset with dummy images
├── attendance_log.csv      # Attendance log file
├── face_database.pkl       # Pre-computed face embeddings
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
├── README.md               # Project documentation
└── .gitignore              # Ignored files & folders
```


---

## 🎥 Demo Video -> [Link](https://drive.google.com/file/d/1Pw2cnxYZ4aPXzW5CsXZ3YNk9lnjLMrRK/view?usp=sharing)
---

## 🔧 Installation & Setup

### **1. Clone the Repository**
```
git clone https://github.com/dhruvsharma2311/face-detection-web-app.git
cd face-detection-web-app
```

### **2. Create a Virtual Environment**
```
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```
pip install --upgrade pip
pip install -r requirements.txt
```

### **4. Run the Application**
Since dummy data is already included in `dataset/`, you can run the app immediately:
```
streamlit run app.py
```

### **5. Access the Web App**
Once Streamlit starts, it will open your browser automatically.  
If it doesn’t, manually visit:
```
http://localhost:8501
```

---

## 📌 Notes

- Virtual environments (`venv/`) are ignored in `.gitignore` and should not be committed.  
- The included dataset is **for demonstration only** — replace with your own images for real use.  
- Works best with **clear, front-facing** images.

---

## 📜 License
MIT License © 2025 Dhruv Sharma
