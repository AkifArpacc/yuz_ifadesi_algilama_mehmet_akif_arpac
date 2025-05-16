#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("model.pkl", "rb") as f:
   model = pickle.load(f)

def draw_landmarks_on_image(rgb_image, detection_result):
  # bulunna yüzler ve o yüzler üzerindeki koordinatlar
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # print("Bulunan yüz sayısı", len(face_landmarks_list))
 
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    # print("Nokta sayısı", len(face_landmarks))

    # Sadece x,y ve z koordinatlarını al
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    # print(len([landmark.x for landmark in face_landmarks]))
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    koordinatlar= []
    for landmark in face_landmarks:
       koordinatlar.append(round(landmark.x, 4))
       koordinatlar.append(round(landmark.y,4))

    sonuc = model.predict([koordinatlar])
    annotated_image = cv2.putText(annotated_image, 
                                  sonuc[0], 
                                  (60,60),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                    2,
                                    (255, 255, 0),
                                    8)


  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()




etiket = "happy"   



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Kameradan görüntü alımı
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)       
        detection_result = detector.detect(mp_image)        
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("yuz", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            exit(0)        
