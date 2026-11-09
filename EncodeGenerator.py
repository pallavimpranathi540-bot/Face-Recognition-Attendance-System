
import os
import pickle
import numpy as np
import face_recognition

# Folder containing student photos
path = "Images"

student_ids = []
images = []

print("NumPy version:", np.__version__)
print("Loading student images...")

# Load images directly as RGB
for file_name in sorted(os.listdir(path)):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    file_path = os.path.join(path, file_name)

    try:
        image = face_recognition.load_image_file(file_path)

        # Ensure a valid, contiguous 8-bit RGB image
        image = np.asarray(image, dtype=np.uint8)
        image = np.ascontiguousarray(image)

        if image.ndim != 3 or image.shape[2] != 3:
            print(f"Skipping {file_name}: invalid image shape {image.shape}")
            continue

        images.append(image)
        student_ids.append(os.path.splitext(file_name)[0])

    except Exception as error:
        print(f"Could not load {file_name}: {error}")

print("Number of images:", len(images))


def find_encodings(images):
    encode_list = []

    for index, image in enumerate(images):
        student_id = student_ids[index]

        print(
            f"\nProcessing {student_id}: "
            f"shape={image.shape}, "
            f"dtype={image.dtype}, "
            f"contiguous={image.flags['C_CONTIGUOUS']}"
        )

        try:
            # Detect faces
            face_locations = face_recognition.face_locations(
                image, model="hog"
            )

            print("Faces detected:", len(face_locations))

            if len(face_locations) != 1:
                print(
                    f"Skipping {student_id}: "
                    f"expected 1 face, found {len(face_locations)}"
                )
                encode_list.append(None)
                continue

            # Generate face encoding
            face_encodings = face_recognition.face_encodings(
                image, face_locations
            )

            if not face_encodings:
                print(f"Skipping {student_id}: encoding failed")
                encode_list.append(None)
                continue

            encode_list.append(face_encodings[0])
            print(f"Successfully encoded: {student_id}")

        except Exception as error:
            print(f"Error processing {student_id}: {error}")
            raise

    return encode_list


# Generate encodings
encodings = find_encodings(images)

# Keep only successfully encoded students
valid_data = [
    (student_ids[i], encodings[i])
    for i in range(len(encodings))
    if encodings[i] is not None
]

if not valid_data:
    print("\nNo valid face encodings generated.")
    print("EncodeFile.p has not been overwritten.")
    raise SystemExit(1)

# Save encodings
with open("EncodeFile.p", "wb") as file:
    pickle.dump(valid_data, file)

print("\nEncoding completed!")
print("Students encoded:", len(valid_data))
print("Saved to EncodeFile.p")
