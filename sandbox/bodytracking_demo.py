import cv2
import mediapipe as mp
import time
import math
import pyvista as pv
from arap2 import arap
import numpy as np
import time

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                smooth_landmarks=self.smooth,
                                min_detection_confidence=self.detectionCon,
                                min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        #print(cTime, self.pTime)
        fbs = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fbs)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # some time this angle comes zero, so below conditon we added
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 1)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 1)
            # cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 2)
        return angle

def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)



import cProfile, pstats #TODO profile


def adjust_point(addedspheres, bunnyarap, body_positions, plotter, pr, mesh, r):
    
    constrains = []

    mesh_point = {
        'left_hand': 155,
        'right_hand': 395,
        'left_thumb': 192,
        'right_thumb': 515,
        'left_pinky': 271,
        'right_pinky': 432,
        'left_elbow': 75,
        'right_elbow': 321,
        'left_eye': 6,
        'right_eye': 9,
        'left_mouth': 10,
        'right_mouth': 15,
        'left_shoulder': 52,
        'right_shoulder': 301,
        'left_hip': 49,
        'right_hip': 299,
        'left_knee': 111,
        'right_knee': 364,
        'left_ankle': 118,
        'right_ankle': 365,
    }

    scale=r*0.003 # 0.003 for the dance video, 0.001 for live cam(upper half body)
    for key, value in body_positions.items():
        i, x, y = value
        scaled_val = np.array([x, -y, 0]) * scale
        point_num = mesh_point[key]
        constrains.append((point_num, scaled_val))

    
    for actor in addedspheres:#remove the old red spheres
        plotter.remove_actor(actor,render=False)
    addedspheres.clear()
    for i, point in constrains:#add the new red spheres
        sphere = pv.Sphere(radius=r * 0.01, center=point)
        addedspheres.append(plotter.add_mesh(sphere, color='red',render=False))
    

    bunnyarap.setconstraints(constrains)
    pr.enable()
    P_ = bunnyarap.apply()
    pr.disable()
    stats = pstats.Stats(pr)
    #stats.strip_dirs().sort_stats('tottime').print_stats(15)

    mesh.points = P_
    plotter.update()


    


def main():

    # Arap part
    meshpath = "../resources/meshes/BunnyLowPoly.stl"
    meshpath = "./resources/meshes/bunny.obj"
    meshpath = "../resources/meshes/lowpoly_male.obj"
    mesh = pv.read(meshpath)
    mesh.clean(inplace=True)
    print("imported mesh")

    r = getmaxbound(mesh)
    bunnyarap = arap(mesh)

    plotter = pv.Plotter()

    plotter.add_mesh(mesh, show_edges=True)
    plotter.set_background('black')
    plotter.add_axes(color="white")
    plotter.show(interactive_update=True)
    print("plotter initialized")
    addedspheres = []
    pr = cProfile.Profile(builtins=False)

    # Body tracking Part

    detector = poseDetector()
    print("detector initialized")

    LIVE_CAM = False
    video_path = '../resources/videos/dance.mp4'

    if LIVE_CAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    print("cam initialized")

    cnt = 0

    body_index = {
        'left_hand': 15,
        'right_hand': 16,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_eye': 3,
        'right_eye': 6,
        'left_mouth': 9,
        'right_mouth': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.getPosition(img)

            if len(lmList) < 1:
                continue

            body_positions = {}
            for key, value in body_index.items():
                _, x, y = lmList[value]

                # Use body parts only if the part was actually recognized inside the frame.
                if x < img.shape[1] and y < img.shape[0]:
                    body_positions[key] = lmList[value]


            print(body_positions)
            adjust_point(addedspheres, bunnyarap, body_positions, plotter, pr, mesh,r)

            detector.showFps(img)
            cv2.imshow("Image", img)
            if cv2.waitKey(1)==ord("q"):
                break

    cap.release()



if __name__ == "__main__":
    main()
