from djitellopy import Tello
from pupil_apriltags import Detector
from task import Task1, Task2, Task3
from demo import Yolo
import cv2
import numpy as np
import time
FX = 922.3494152773385
FY = 918.5890942187204
CX = 480.8208635422829
CY = 374.0898996576405
DIST_COEFF = np.array([ 6.53506073e-02 ,-8.58693898e-01 ,-9.16520050e-04, 2.32928669e-04,2.94755940e+00])

INTRINSIC = np.array([[ FX,  0, CX],
                        [  0, FY, CY],
                        [  0,  0,  1]])
TagSize = 0.08
T3_distance = 50
def main():
    drone = Tello()
    drone.connect()
    print(drone.get_battery())
    drone.streamon()
    detector = Detector(families="tag36h11")
    drone.takeoff()
    time.sleep(0.2)
    drone.move_up(50)
    stop_cnt=0
    task1 = Task1(drone)
    task1_finished = False
    task2 = Task2(drone, TagSize)
    task2_finished = False
    task3 = Task3(drone, TagSize)
    task3_finished = False
    yolo = Yolo()
    yolo_task_finished =False

    all_task_done = False
    while True:
        stop_cnt+=1
        if stop_cnt == 100:
            drone.send_command_with_return("stop")
            stop_cnt = 0
        frame = drone.get_frame_read().frame
        tag_list = detector.detect(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=[FX,FY,CX,CY],
            tag_size=TagSize,
        )

        if not task1_finished:
            task1_finished = task1.run(tag_list, frame)

            if task1_finished:
                time.sleep(1.5)
                print("task 1 done")
        elif not task2_finished:
            task2_finished = task2.run(tag_list, frame ,task1.direct)

            if task2_finished:
                up_down = -task2.distance[1]-0.2*task2.distance[2]+45
                if task1.direct=="Right":
                    if abs(int(task2.distance[0]))>20:
                        drone.move_right(abs(int(task2.distance[0])))
                    if up_down>0 and abs(up_down)>20:
                        drone.move_up(int(up_down))
                    elif up_down<0 and abs(up_down)>20:
                        drone.move_down(abs(int(up_down)))
                    drone.move_forward(int(task2.distance[2])+50)
                    #回正
                    if abs(int(task2.distance[0]))>20:
                        drone.move_left(abs(int(task2.distance[0])))
                elif task1.direct =="Left":
                    if abs(int(task2.distance[0]))>20:
                        drone.move_left(abs(int(task2.distance[0])))
                    if up_down>0 and abs(up_down)>20:
                        drone.move_up(int(up_down))
                    elif up_down<0 and abs(up_down)>20:
                        drone.move_down(abs(int(up_down)))
                    drone.move_forward(int(task2.distance[2])+50)
                    #回正
                    if abs(int(task2.distance[0]))>20:
                        drone.move_right(abs(int(task2.distance[0])))
                time.sleep(2)
                print("task 2 done")
                drone.move_forward(T3_distance)
                drone.move_up(20)
        elif not yolo_task_finished:
            yolo_task_finished = yolo.run(frame)
            if yolo_task_finished:
                drone.move_back(T3_distance)
        #     while True:
        #         key = cv2.waitKey(20)
        #         if (key & 0xFF) == ord("3") or(key & 0xFF) == ord("4")or(key & 0xFF) == ord("5")or(key & 0xFF) == ord("6"):
        #             tg = (key & 0xFF)
        #             yolo_task_finished = True
        #             break
        elif not task3_finished:
            
            task3_finished = task3.run(tag_list, frame ,yolo.label)
            if task3_finished:
                if task3.distance[0]>=0:
                    if abs(int(task3.distance[0]))>20:
                        drone.move_right(abs(int(task3.distance[0])))
                    drone.move_forward(int(task3.distance[2])-10)
                else:
                    if abs(int(task3.distance[0]))>20:
                        drone.move_left(abs(int(task3.distance[0])))
                    drone.move_forward(int(task3.distance[2])-10)
            

        for tag in tag_list:
            for corner in tag.corners:
                cv2.circle(frame, list(map(int, corner)), 5, (0, 0, 255), -1)

        cv2.imshow("drone", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(30)
        if (key & 0xFF) == ord("q") or all_task_done:
            drone.land()
            drone.streamoff()
            break

    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
