import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

##Cambiar static_image_mode = False para aplicarlo a vídeo y no a imagenes
with mp_pose.Pose(
    static_image_mode=True) as pose:

    image = cv2.imread("prueba3.jpg")
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    print("Pose landmarks: ", results.pose_landmarks)

    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=6))
    
    ##plt.imshow(image_rgb)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()