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


def adjust_point(addedspheres, bunnyarap, right, left, plotter, pr, mesh, r):
    
    constrains = []


    #if sum(right) > 15:
    #    constrains.append((17, bunnyarap.P[17] + np.array([right[0], right[1], 0]) * r * 0.001))
    #if sum(left) > 15:
    #    constrains.append((23, bunnyarap.P[23] + np.array([left[0], left[1], 0]) * r * 0.001))
    scale=r*0.001

    i,x,y=left#set left constraint
    left=np.array([x,y,0])*scale
    constrains.append((517,left))

    i,x,y=right#set right constraint
    right=np.array([x,y,0])*scale
    constrains.append((1309, right))

    
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
    stats.strip_dirs().sort_stats('tottime').print_stats(15)

    mesh.points = P_
    plotter.update()


    


def main():

    # Arap part
    meshpath = "../resources/meshes/BunnyLowPoly.stl"
    meshpath = "./resources/meshes/bunny.obj"
    meshpath = "./resources/meshes/lowpoly_male.obj"
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
    cap = cv2.VideoCapture(0)
    print("cam initialized")
    left_hand_prev = [0, 0, 0]
    right_hand_prev = [0, 0, 0]
    cnt = 0
    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.getPosition(img)
            #print(f'right hip : {lmList[24]}')
            left_hand_cur = lmList[15]
            right_hand_cur = lmList[16]
            if cnt > 0: #True: #(lmList.size > 1):
                print(f'left hand: {left_hand_cur}, right hand: {right_hand_cur}')
                #left_move = [left_hand_cur[1] - left_hand_prev[1], left_hand_cur[2] - left_hand_prev[2]]
                #right_move = [right_hand_cur[1] - right_hand_prev[1], right_hand_cur[2] - right_hand_prev[1]]
                adjust_point(addedspheres, bunnyarap, right_hand_cur, left_hand_cur, plotter, pr, mesh,r)
                

            detector.showFps(img)
            cv2.imshow("Image", img)
            if cv2.waitKey(1)==ord("q"):
                break
            cnt+=1
            


            left_hand_prev = left_hand_cur
            right_hand_prev = right_hand_cur
    cap.release()



if __name__ == "__main__":
    main()
