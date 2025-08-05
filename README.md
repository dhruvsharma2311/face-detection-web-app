# 👁️‍🗨️ Face Recognition Attendance System

A smart web-based attendance system that uses face recognition to identify individuals from group photos and mark attendance automatically. Built using **Python**, **Streamlit**, **InsightFace**, and **OpenCV**.

---

## 🚀 Features

- 📸 Upload group photos to detect and recognize faces
- 🧠 Uses **InsightFace** for accurate face detection & embedding
- 🧮 Cosine similarity for face matching from database
- 🗂️ Automatically tracks present, absent, and unknown individuals
- 🧾 Generates downloadable attendance summary (CSV)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **Face Detection & Recognition**: InsightFace, OpenCV
- **Others**: NumPy, Pandas

---

## 🖥️ Demo

<img src="demo_screenshot.png" width="600">

> _Optional: Add a GIF or video link showing the workflow in action._

---

## 📁 Folder Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Project dependencies
├── dataset/                # User images organized in folders (ID_Name)
├── output/                 # Attendance CSVs and logs
```

---

## 🔧 Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/face-attendance-app.git
cd face-attendance-app
```

2. **Install dependencies**
``` bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```
