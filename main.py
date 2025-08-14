# main.py

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


def load_face_analyzer():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def build_database(dataset_path='dataset', database_file='face_database.pkl'):
    face_db = {}
    face_app = load_face_analyzer()

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            embeddings = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = face_app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                face_db[person_folder] = avg_embedding

    with open(database_file, 'wb') as f:
        pickle.dump(face_db, f)

    print(f"✅ Database built with {len(face_db)} people.")


def add_person_to_database(face_app, dataset_path='dataset', database_file='face_database.pkl'):
    new_id = input("Enter new person's ID: ").strip()
    new_name = input("Enter new person's Name: ").strip()
    folder_name = f"{new_id}_{new_name}"
    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"✅ Created folder: {folder_path}")
    input("🛑 Add the person's images to the folder and press Enter to continue...")

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_paths:
        print("❌ No images found.")
        return

    if os.path.exists(database_file):
        with open(database_file, 'rb') as f:
            face_db = pickle.load(f)
    else:
        face_db = {}

    embeddings = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        face_db[folder_name] = avg_embedding
        with open(database_file, 'wb') as f:
            pickle.dump(face_db, f)
        print(f"✅ Added {folder_name} to database.")
    else:
        print("❌ No valid faces detected.")


def recognize_faces(face_app, database_file='face_database.pkl'):
    group_photo_path = input("Enter the group photo filename (e.g., grp_photo5.png): ").strip()

    if not os.path.exists(group_photo_path):
        print("❌ Group photo not found.")
        return

    with open(database_file, 'rb') as f:
        face_db = pickle.load(f)

    known_ids = list(face_db.keys())
    known_embeddings = np.stack(list(face_db.values()))

    img = cv2.imread(group_photo_path)
    group_faces = face_app.get(img)
    print(f"✅ Detected {len(group_faces)} faces in group photo.")

    attendance_records = []
    detected_ids = set()
    threshold = 0.6

    for i, face in enumerate(group_faces):
        embedding = face.embedding.reshape(1, -1)
        similarities = cosine_similarity(embedding, known_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        print(f"\nFace #{i+1} | Similarity: {best_score:.3f}")

        if best_score > threshold:
            matched_id = known_ids[best_match_idx]
            print(f"✅ Matched with: {matched_id}")
            attendance_records.append((matched_id, "Present"))
            detected_ids.add(matched_id)
        else:
            attendance_records.append(("Unknown", "Unknown"))

    for id_name in known_ids:
        if id_name not in detected_ids:
            attendance_records.append((id_name, "Absent"))

    df = pd.DataFrame(attendance_records, columns=["ID_Name", "Status"])
    df[['ID', 'Name']] = df['ID_Name'].str.split('_', expand=True)
    df = df[['ID', 'Name', 'Status']]
    df.to_csv('attendance.csv', index=False)
    print("✅ Attendance saved to attendance.csv")



if __name__ == "__main__":
    face_app = load_face_analyzer()

    print("\n1. Build database from dataset")
    print("2. Add a new person to database")
    print("3. Recognize faces from group photo")
    choice = input("Select option (1/2/3): ").strip()

    if choice == '1':
        build_database()
    elif choice == '2':
        add_person_to_database(face_app)
    elif choice == '3':
        recognize_faces(face_app)
    else:
        print("❌ Invalid choice.")
