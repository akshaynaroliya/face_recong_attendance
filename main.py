import face_recognition
import cv2
import csv
from datetime import datetime

# Define the known faces with names and image paths
known_faces = [
    {"name": "Akshay", "image_path": "faces/akshay.jpg"},
    {"name": "Rohan", "image_path": "faces/rohan.jpg"}
]

# Initialize lists to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known face encodings and names
for face in known_faces:
    image = face_recognition.load_image_file(face["image_path"])
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(face["name"])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Check if video capture was successful
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read a frame from the video source.")
        break
    
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each face in the frame
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distance.argmin()

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Get the current date and time
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            csv_filename = f"{current_date}.csv"

            # Write to the CSV file
            with open(csv_filename, "a+", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([name, now.strftime("%H:%M:%S")])

            # Draw a box and label around the face
            top, right, bottom, left = face_locations[best_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Attendance", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
