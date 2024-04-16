import cv2
import mediapipe as mp
from calculate_angle import calculate_angle
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("./videos/sentadillamalatecnicabeto.mp4")
# Obtener las dimensiones originales del video
original_width = int(cap.get(3))
original_height = int(cap.get(4))

# Redimensionar el video para que tenga 450 píxeles de ancho
new_width = 450
aspect_ratio = new_width / original_width
dim = (new_width, int(original_height * aspect_ratio))

# Crear el directorio si no existe
output_directory = "./videos_results"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Inicializar el VideoWriter con las dimensiones redimensionadas
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_path = os.path.join(output_directory, 'resultadoSentadillaMalaTecnicaBeto.avi')
out = cv2.VideoWriter(out_path, fourcc, 20.0, dim)


with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, dim)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            # Usa las dimensiones redimensionadas para calcular x1 y y1
            x1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * dim[0])
            y1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * dim[1])

            x2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * dim[0])
            y2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * dim[1])

            x3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * dim[0])
            y3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * dim[1])

            x4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * dim[0])
            y4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * dim[1])

            x5 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * dim[0])
            y5 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * dim[1])

            x6 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * dim[0])
            y6 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * dim[1])

            x7 = int(results.pose_landmarks.landmark[31].x * dim[0])
            y7 = int(results.pose_landmarks.landmark[31].y * dim[1])

            
            angle2 = calculate_angle([x1, y1], [x2, y2], [x3, y3])
            
            # Determinar el color basado en el valor del ángulo
            if angle2 < 180 :
                line_color = (0, 0, 255)  # Verde
                cv2.putText(frame, "Mal hecho", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                line_color = (0, 255, 0)  # Rojo

            cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)
            cv2.line(frame, (x2, y2), (x3, y3), line_color, 3)
            cv2.line(frame, (x3, y3), (x4, y4), line_color, 3)
            cv2.line(frame, (x4, y4), (x5, y5), line_color, 3)
            cv2.line(frame, (x5, y5), (x6, y6), line_color, 3)
            cv2.line(frame, (x6, y6), (x7, y7), line_color, 3)

            cv2.circle(frame, (x1, y1), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x3, y3), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x4, y4), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x5, y5), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x6, y6), 15, (200, 0, 0), -1)
            cv2.circle(frame, (x7, y7), 15, (200, 0, 0), -1)

            angle1 = calculate_angle([x3, y3], [x4, y4], [x5, y5])
            cv2.putText(frame, str(int(angle1)), (x4 - 50, y4), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            cv2.putText(frame, str(int(angle2)), (x2 - 50, y2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            angle3 = calculate_angle([x5, y5], [x6, y6], [x7, y7])
            cv2.putText(frame, str(int(angle3)), (x6 - 50, y6), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
cap.release()
out.release()
cv2.destroyAllWindows()