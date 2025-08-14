import os
import cv2
import numpy as np
import pandas as pd
import pickle
import datetime
import zipfile
import streamlit as st
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# Set page configuration first
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Load InsightFace model
@st.cache_resource
def load_face_analyzer():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = load_face_analyzer()

# Load face database
def load_database(database_file='face_database.pkl'):
    if os.path.exists(database_file):
        with open(database_file, 'rb') as f:
            return pickle.load(f)
    return {}

# Automatically extract dataset from `dataset.zip` if available and build the database
def load_and_build_database(dataset_zip='dataset.zip', database_file='face_database.pkl'):
    # Check if dataset.zip exists
    if os.path.exists(dataset_zip):
        # Unzip the dataset if not already extracted
        if not os.path.exists('dataset'):
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall('dataset')
            st.info("✅ Dataset extracted successfully from dataset.zip")
        else:
            st.info("✅ Dataset already extracted.")
    
    # Load or initialize face database
    face_db = load_database(database_file)
    
    # Process all images in the dataset folder to build the database
    if not face_db:
        st.info("Building the face database...")
        for person_folder in os.listdir('dataset'):
            person_path = os.path.join('dataset', person_folder)
            if os.path.isdir(person_path):
                embeddings = []
                for file in os.listdir(person_path):
                    img_path = os.path.join(person_path, file)
                    img = cv2.imread(img_path)
                    faces = face_app.get(img)
                    if faces:
                        embeddings.append(faces[0].embedding)

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    normalized_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                    face_db[person_folder] = normalized_embedding

        with open(database_file, 'wb') as f:
            pickle.dump(face_db, f)
        st.success("✅ Face database built successfully!")

    return face_db

# Recognize faces in group photo
def recognize_faces_from_group(image, face_app, face_db):
    known_ids = list(face_db.keys())
    known_embeddings = np.stack(list(face_db.values()))

    group_faces = face_app.get(image)
    st.info(f"✅ Detected {len(group_faces)} faces in group photo.")

    attendance_records = []
    detected_ids = set()
    threshold = 0.6

    for i, face in enumerate(group_faces):
        
        embedding = face.embedding / np.linalg.norm(face.embedding)
        embedding = embedding.reshape(1, -1)

        similarities = cosine_similarity(embedding, known_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score > threshold:
            matched_id = known_ids[best_match_idx]
            detected_ids.add(matched_id)
            id_split = matched_id.split('_')
            attendance_records.append({
                'ID': id_split[0],
                'Name': id_split[1],
                'Status': 'Present',
                'Similarity': round(best_score, 3),
                'Photo': image[face.bbox.astype(int)[1]:face.bbox.astype(int)[3],
                               face.bbox.astype(int)[0]:face.bbox.astype(int)[2]]
            })
        else:
            attendance_records.append({
                'ID': 'Unknown',
                'Name': 'Unknown',
                'Status': 'Unknown',
                'Similarity': round(best_score, 3),
                'Photo': image[face.bbox.astype(int)[1]:face.bbox.astype(int)[3],
                               face.bbox.astype(int)[0]:face.bbox.astype(int)[2]]
            })

    for known_id in known_ids:
        if known_id not in detected_ids:
            id_split = known_id.split('_')
            attendance_records.append({
                'ID': id_split[0],
                'Name': id_split[1],
                'Status': 'Absent',
                'Similarity': '',
                'Photo': None
            })

    return pd.DataFrame(attendance_records)

# Add new person with automatically generated ID
def add_person(face_app, new_name, uploaded_images, dataset_path='dataset', database_file='face_database.pkl'):
    # Load existing database
    if os.path.exists(database_file):
        with open(database_file, 'rb') as f:
            face_db = pickle.load(f)
    else:
        face_db = {}

    # Generate the next available ID
    existing_ids = list(face_db.keys())
    if existing_ids:
        # Extract numeric part of the IDs (e.g., ID001, ID002, etc.)
        existing_numbers = [int(id.split('_')[0][2:]) for id in existing_ids]
        next_id_number = max(existing_numbers) + 1  # Get next ID number
    else:
        next_id_number = 1  # If no IDs exist, start from 1

    new_id = f"ID{next_id_number:03d}"  # Generate new ID (e.g., ID001, ID002)

    folder_name = f"{new_id}_{new_name}"
    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save uploaded images
    for i, uploaded_img in enumerate(uploaded_images):
        img_bytes = uploaded_img.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        save_path = os.path.join(folder_path, f"{i}.jpg")
        cv2.imwrite(save_path, img)

    # Build embeddings
    embeddings = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        faces = face_app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if embeddings:
        # L2 normalization (also known as Euclidean normalization) is a technique used
        # to scale a vector so that its length (or magnitude) becomes exactly 1,
        # but the direction remains the same.
        avg_embedding = np.mean(embeddings, axis=0)
        normalized_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        face_db[folder_name] = normalized_embedding


        with open(database_file, 'wb') as f:
            pickle.dump(face_db, f)

        return new_id  # Return the generated ID
    return None

# Streamlit App
st.title("🧑‍🏫 Face Recognition Attendance System")

tab1, tab2 = st.tabs(["📸 Upload Group Photo", "➕ Add New Person"])

# -------------------- TAB 1: Attendance ---------------------
with tab1:
    uploaded_group = st.file_uploader("Upload a group photo:", type=["jpg", "jpeg", "png"])
    if uploaded_group:
        file_bytes = np.asarray(bytearray(uploaded_group.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Group Photo", use_column_width=True)

        if st.button("🔍 Process Attendance"):
            face_db = load_and_build_database()
            df = recognize_faces_from_group(image, face_app, face_db)

            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            filename_timestamp = now.strftime("%Y%m%d_%H%M%S")

            df['Timestamp'] = timestamp

            # Append current session to persistent attendance log
            log_file = "attendance_log.csv"
            if os.path.exists(log_file):
                existing_log = pd.read_csv(log_file)
                combined_log = pd.concat([existing_log, df], ignore_index=True)
            else:
                combined_log = df

            combined_log.to_csv(log_file, index=False)

            st.success("✅ Attendance Processed!")

            for idx, row in df.iterrows():
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    if row['Photo'] is not None:
                        st.image(row['Photo'], width=100)
                    else:
                        st.image("https://via.placeholder.com/100", caption="No Image", width=100)
                with col2:
                    st.markdown(f"**ID:** {row['ID']}")
                    st.markdown(f"**Name:** {row['Name']}")
                with col3:
                    if row['Status'] == "Present":
                        st.markdown("✅ **Present**")
                    elif row['Status'] == "Absent":
                        st.markdown("❌ **Absent**")
                    else:
                        st.markdown("⚠️ **Unknown**")

            with st.expander("📋 View Full Attendance Table"):
                st.dataframe(df[['ID', 'Name', 'Status', 'Similarity', 'Timestamp']])

                csv = df[['ID', 'Name', 'Status', 'Similarity', 'Timestamp']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Attendance CSV",
                    data=csv,
                    file_name=f"attendance_{filename_timestamp}.csv",
                    mime='text/csv'
                )

            # Compute attendance summary
            summary_df = combined_log[combined_log['Status'] != 'Unknown']
            attendance_summary = summary_df.groupby(['ID', 'Name'])['Status'].value_counts().unstack(fill_value=0).reset_index()
            attendance_summary['Total Sessions'] = attendance_summary.get('Present', 0) + attendance_summary.get('Absent', 0)
            attendance_summary['Attendance (%)'] = (attendance_summary.get('Present', 0) / attendance_summary['Total Sessions'] * 100).round(2)

            with st.expander("📊 View Attendance Summary Across Sessions"):
                st.dataframe(attendance_summary)

                summary_csv = attendance_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Attendance Summary CSV",
                    data=summary_csv,
                    file_name="attendance_summary.csv",
                    mime='text/csv'
                )

# -------------------- TAB 2: Add New Person ---------------------
with tab2:
    st.subheader("Add a New Person to the Dataset")
    new_name = st.text_input("Enter Name")
    uploaded_images = st.file_uploader("Upload 5-8 face images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if st.button("➕ Add to Database"):
        if new_name and uploaded_images:
            new_id = add_person(face_app, new_name.strip(), uploaded_images)
            if new_id:
                st.success(f"✅ {new_id} - {new_name} added to database!")
            else:
                st.error("❌ Failed to detect valid faces in uploaded images.")
        else:
            st.warning("⚠️ Please provide Name and upload images.")
