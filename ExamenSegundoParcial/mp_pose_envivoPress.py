import cv2
import mediapipe as mp
from calculate_angle import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose(
  static_image_mode = False) as pose:

  while True:
    ret, frame = cap.read()
    if ret == False:
      break
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
      # Usa las dimensiones redimensionadas para calcular x1 y y1
            x1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
            y1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

            x2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width)
            y2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)

            x3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width)
            y3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

            x4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * width)
            y4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height)

            angle = calculate_angle([x1, y1], [x2, y2], [x3, y3])

            # Establecer el color de las líneas según el ángulo
            if 12 <= angle <= 30:
                line_color = (0, 255, 0)  # Verde
                cv2.putText(frame, "Bien hecho", (width//2, height//4), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                line_color = (0, 0, 255)  # Rojo

            cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)
            cv2.line(frame, (x2, y2), (x3, y3), line_color, 3)
            cv2.line(frame, (x1, y1), (x4, y4), line_color, 3)

            cv2.circle(frame, (x1, y1), 6, (200, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 6, (200, 0, 0), -1)
            cv2.circle(frame, (x3, y3), 6, (200, 0, 0), -1)
            cv2.circle(frame, (x4, y4), 6, (200, 0, 0), -1)

            cv2.putText(frame, str(int(angle)), (x2 - 50, y2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
      
    cv2.imshow("Frame", frame)
    ##plt.imshow(frame)
    if cv2.waitKey(1) & 0xFF == 27:
      break
    
cap.release()
cv2.destroyAllWindows()