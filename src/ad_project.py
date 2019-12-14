#!/usr/bin/env python

import cv2, time, rospy, rospkg, math, os, numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy, Image
from std_msgs.msg import Int32MultiArray

from fillter import *
from motordriver import *
from obstacledetector import *
from traffic_detect import *

bridge = CvBridge()
pub = None
cv_image = np.empty(shape=[0])
gstart_time = None
g_stop = False


class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]

    def get_mm(self):
        return float(sum(self.data)) / len(self.data)

    def get_wmm(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

class ObstacleDetector:

    def __init__(self, topic):
        self.left = -1
        self.mid = -1
        self.right = -1
        rospy.Subscriber(topic, Int32MultiArray, self.read_distance)

    def read_distance(self, data):
        self.left = data.data[0]
        self.mid = data.data[1]
        self.right = data.data[2]

    def get_distance(self):
        return self.left, self.mid, self.right


def img_callback(data):
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as e:
        rospy.loginfo(e)


def auto_drive(Angle, Speed):
    global pub
    drive_info = [
     Angle, Speed]
    drive_info = Int32MultiArray(data=drive_info)
    pub.publish(drive_info)

def file_count(filePath):
    return len(os.walk(filePath).next()[2])


def detect_line(edge_img, hsv_mask):
    col = 640
    rpos = 0
    thresh_pixel_cnt = 30
    height_offset = 15
    r_width = 20
    r_height = 10
    i = height_offset
    for rcol in range(col / 2, col):
        if rcol >= col / 2 and rcol <= col:
            if edge_img[(i, rcol - 1)] == 255:
                detect_area = hsv_mask[i - r_height:i, rcol - 1:rcol - 1 + r_width]
                nonzero = cv2.countNonZero(detect_area)
                if nonzero > thresh_pixel_cnt:
                    rpos = rcol - 1
                    break

    return rpos


def detect_yellowline(frame):
    width = 640
    offset_roi = 125
    mask = frame[430 - offset_roi:450 - offset_roi, 0:width]
    lower_white = np.array([120, 200, 200], dtype=np.uint8)
    upper_white = np.array([160, 240, 255], dtype=np.uint8)

    mask = cv2.inRange(mask, lower_white, upper_white)

    if cv2.countNonZero(mask) > 100:
        print(cv2.countNonZero(mask))

    if cv2.countNonZero(mask) >= 300:
        return True
    return False


def process_image(frame, brightness):
    image_width = 640
    image_height = 480
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width = 640
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    offset_roi = 125
    mask = frame[430 - offset_roi:450 - offset_roi, 0:width]
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, brightness], dtype=np.uint8)
    upper_white = np.array([131, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    edge2_img = edge_img[430 - offset_roi:450 - offset_roi, 0:width]
    hsv_mask = mask

    lines = cv2.HoughLinesP(edge2_img, 1, 1 * np.pi / 180, 30, np.array([]), minLineLength=15, maxLineGap=50)

    l_x1_s, l_y1_s, l_x2_s, l_y2_s = 0, 0, 0, 0
    r_x1_s, r_y1_s, r_x2_s, r_y2_s = 0, 0, 0, 0

    l_x1_a, l_y1_a, l_x2_a, l_y2_a = 0, 0, 0, 0
    r_x1_a, r_y1_a, r_x2_a, r_y2_a = 0, 0, 0, 0

    left_num = 0
    right_num = 0

    left_slope = 0
    right_slope = 0

    copy_img = np.array(frame)
    copy_img = copy_img[430 - offset_roi:450 - offset_roi, 0:width]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]


            if int(x1 - x2) == 0:
                continue

            if (y2-y1) < 0 and (x1 < 300 and x2 < 300):
                l_x1_s += x1
                l_y1_s += y1
                l_x2_s += x2
                l_y2_s += y2
                left_num += 1
                cv2.line(copy_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            elif (y2-y1) > 0 and (x1 > 340 and x2 > 340):
                r_x1_s += x1
                r_y1_s += y1
                r_x2_s += x2
                r_y2_s += y2
                right_num += 1
                cv2.line(copy_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if left_num != 0:
        l_x1_a = l_x1_s / left_num
        l_y1_a = l_y1_s / left_num
        l_x2_a = l_x2_s / left_num
        l_y2_a = l_y2_s / left_num

    if right_num != 0:
        r_x1_a = r_x1_s / right_num
        r_y1_a = r_y1_s / right_num
        r_x2_a = r_x2_s / right_num
        r_y2_a = r_y2_s / right_num

    if left_num == 0:
        l_x1_a, l_y1_a, l_x2_a, l_y2_a = 0, 0, 0, 0

    if right_num == 0:
        r_x1_a, r_y1_a, r_x2_a, r_y2_a = 0, 0, 0, 0


    if (l_x2_a - l_x1_a) != 0:
        left_slope = (l_y2_a - l_y1_a) / (l_x1_a - l_x2_a)

    if (r_x2_a - r_x1_a) != 0:
        right_slope = (r_y2_a - r_y1_a) / (r_x1_a - r_x2_a)



    if (l_x2_a - l_x1_a) == 0:
        left_slope = 9999

    if (r_x2_a - r_x1_a) == 0:
        right_slope = 9999


    #cv2.imshow('edge', edge2_img)
    #cv2.imshow("lines", copy_img)

    return (left_slope, right_slope, l_x1_a, l_x2_a, r_x1_a, r_x2_a)

def steer(lslope, rslope, l_x1_a, l_x2_a, r_x1_a, r_x2_a):
    angle = 0
    y_offset = 240
    x_offset = 50

    default = 170

    # X X
    if lslope >= 9900 and rslope >= 9900:
        return 90

    # / \
    if lslope <= 9900 and rslope <= 9900:
        x_offset = (l_x2_a + r_x2_a) / 2 - 320

    # X \
    if lslope >= 9900 and rslope <= 9900:
        x_offset = r_x1_a - r_x2_a - default

    # / X
    if rslope >= 9900 and lslope <= 9900:
        x_offset = l_x2_a + l_x1_a + default


    radian = math.atan(x_offset / y_offset)

    # radian to degree
    angle = radian / math.pi * 180.0
    angle = int(round(angle))

    return angle + 90


def accelerate(angle):
    speed = 139

    if angle <= 65 or angle >= 115:
        speed = 134

    if 88 <= angle <= 92:
        speed = 140

    return speed

def start(brightness):
    global g_stop
    global pub
    offset = 56
    rospy.init_node('xycar_b2')
    pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
    rospy.loginfo('Xycar B2 KM v2.1')
    image_sub = rospy.Subscriber('/usb_cam/image_raw/', Image, img_callback)
    filePath = '/home/nvidia/xycar/src/xycar_b2/record/' \
               + 'test' + str(file_count('/home/nvidia/xycar/src/xycar_b2/record')) + '.avi'
    recorder = cv2.VideoWriter(
        filePath,
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (640, 480)
    )
    obstacle_detector = ObstacleDetector('/ultrasonic')

    rospy.sleep(2)
    speed = 130
    start_time = time.time()

    leftm = MovingAverage(10)
    midm = MovingAverage(10)
    rightm = MovingAverage(10)

    yellow_flag = False
    yellow_time = 0


    # traffic signs detect
    # Clean previous image
    clean_images()
    # Training phase
    model = training()

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.65  # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []


    while cv_image.size == 921600:
        recorder.write(cv_image)

        obs_l, obs_m, obs_r = obstacle_detector.get_distance()
        leftm.add_sample(obs_l)
        midm.add_sample(obs_m)
        rightm.add_sample(obs_r)


        lslope, rslope, lpos1, lpos2, rpos1, rpos2 = process_image(cv_image, brightness)
        x_location = steer(lslope, rslope, lpos1, lpos2, rpos1, rpos2)
        x_location = int(round(x_location))
        speed = accelerate(x_location)

        if detect_yellowline(cv_image) and (not yellow_flag) and time.time()-start_time >= 30:
            print("Yellow")
            yellow_flag = True
            yellow_time = time.time()

        if yellow_flag and time.time()-yellow_time >= 1:
            break


        if lslope >= 9900 and rslope >= 9900 and x_location == 90:
            speed = 133

        # traffic signs
        coordinate, image, sign_type, text = localization(cv_image, 2, similitary_contour_with_circle, model, count, current_sign)
        if sign_type > 0 and (not current_sign or sign_type != current_sign):
            current_sign = sign_type
            current_text = text


        # 1) Slow
        if current_sign != None and current_text == "SLOW":
            speed = 116

        # 2) Stop
        if current_sign != None and current_text == "STOP":
            speed = 90

        if yellow_flag:
            x_location = (rpos2 - 320 - offset) * 0.6
            x_location = int(round(x_location))

        auto_drive(x_location, speed)

        if cv2.waitKey(1) & 255 == ord('q'):
            break


    cv2.destroyAllWindows()
    recorder.release()

    print("parking")
    for i in range(2):
        auto_drive(90, 90)
        time.sleep(0.1)

    for i in range(12):
        auto_drive(90, 120)
        time.sleep(0.1)

    for i in range(22):
        auto_drive(60, 125)
        time.sleep(0.1)

    for i in range(2):
        auto_drive(90, 90)
        time.sleep(0.1)

    for i in range(28):
        auto_drive(145, 63)
        time.sleep(0.1)

    for i in range(10):
        auto_drive(90, 75)
        time.sleep(0.05)

    rospy.spin()


def main():
    global Angle
    global Speed
    time.sleep(3)
    cfg_path = rospkg.RosPack().get_path('xycar_b2') + '/src/go2.cfg'
    cfg_file = open(cfg_path, 'r')
    brightness = int(cfg_file.readline())
    cfg_file.close()
    Speed = 90
    Angle = 90
    start(brightness)


if __name__ == '__main__':
    main()
# global gstart_time ## Warning: Unused global
# okay decompiling xycar_b2_main_auto.pyc