
import cv2
import pickle
import numpy as np
import face_recognition
import csv
import os
from datetime import datetime, date

# -------------------------------
# 1. Load student face encodings
# -------------------------------
with open("EncodeFile.p", "rb") as file:
    student_data = pickle.load(file)

student_ids = [item[0] for item in student_data]
known_encodings = [item[1] for item in student_data]

print("Registered students:", student_ids)

# -------------------------------
# 2. Set up attendance CSV
# -------------------------------
attendance_file = "Attendance.csv"
today = date.today().isoformat()

# Create the file and its header if needed
if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
    with open(attendance_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Student_ID", "Date", "Time", "Status"])

# Load students already marked present today
attendance_today = set()

with open(attendance_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row.get("Date") == today:
            attendance_today.add(row.get("Student_ID"))

print("Already marked today:", attendance_today)

# -------------------------------
# 3. Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Webcam started. Press Q to quit.")

# -------------------------------
# 4. Recognize faces
# -------------------------------
while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    small_frame = cv2.resize(
        frame, (0, 0), fx=0.25, fy=0.25
    )

    rgb_frame = cv2.cvtColor(
        small_frame, cv2.COLOR_BGR2RGB
    )
    rgb_frame = np.ascontiguousarray(
        rgb_frame, dtype=np.uint8
    )

    face_locations = face_recognition.face_locations(
        rgb_frame, model="hog"
    )
    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations
    )

    for face_location, face_encoding in zip(
        face_locations, face_encodings
    ):
        name = "Unknown"

        if known_encodings:
            distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )
            best_match_index = int(np.argmin(distances))

            if distances[best_match_index] < 0.50:
                name = student_ids[best_match_index]

        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)

        # Draw face rectangle
        cv2.rectangle(
            frame, (left, top), (right, bottom), color, 2
        )

        # Record attendance only once per student per day
        if name != "Unknown" and name not in attendance_today:
            now = datetime.now()

            with open(
                attendance_file, "a",
                newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    name,
                    now.date().isoformat(),
                    now.strftime("%H:%M:%S"),
                    "Present"
                ])

            attendance_today.add(name)
            print(f"Attendance marked: {name} at {now.strftime('%H:%M:%S')}")

        # Display student ID and attendance status
        if name == "Unknown":
            label = "Unknown"
        elif name in attendance_today:
            label = f"{name} - Present"
        else:
            label = name

        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Recognition Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# 5. Close webcam
# -------------------------------
cap.release()
cv2.destroyAllWindows()

print("Face recognition stopped.")
print("Attendance saved in:", attendance_file)
