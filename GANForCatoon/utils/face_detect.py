import cv2
import math
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
import time

class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        # 检测2D图片中的面部标志点
        #time_start = time.time()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)
        #time_end = time.time()
        #print('totally cost', time_end - time_start, 's')

    """
    Dlib所采用的68个人脸关键点标注可以看上图，单边眉毛有5个关键点，从左边界到右边界均匀采样，共5×2=10个。
    眼睛分为6个关键点，分别是左右边界，上下眼睑均匀采样，共6×2=12个。
    嘴唇分为20个关键点，除了嘴角的2个，分为上下嘴唇。上下嘴唇的外边界，各自均匀采样5个点，上下嘴唇的内边界，各自均匀采样3个点，共20个。鼻子的标注增加了鼻梁部分4个关键点，而鼻尖部分则均匀采集5个，共9个关键点。脸部轮廓均匀采样了17个关键点。
    """
    def align(self, image):# 1029 944 3

        landmarks = self.get_max_face_landmarks(image) #（68，2）返回人脸的坐标

        # 绘制关键点坐标
        image_copy = image.copy()
        for i in range(landmarks.shape[0]):
            pt_position = (int(landmarks[i][0]), int(landmarks[i][1]))
            cv2.circle(image_copy, pt_position, 7, (0, 0, 255), -1)
        plt.imshow(image_copy)
        plt.title('face_landmarks')
        plt.show()

        if landmarks is None:
            return None

        else:
            return self.rotate(image, landmarks) # 选择图片，把人脸矫正

    def get_max_face_landmarks(self, image):
        # time_start = time.time()
        preds = self.fa.get_landmarks(image) # (68,2)给出的是landmark坐标   Detect 2D facial landmarks in pictures
        # time_end = time.time()
        # print('totally cost', time_end - time_start, 's')
        if preds is None:
            return None

        elif len(preds) == 1: # 只有一个人脸
            return preds[0] # 返回预测点

        else:
            # find max face 有多张人脸
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    def rotate(self,image, landmarks):# 原图片 关键点
        # rotation angle 旋转的角度
        left_eye_corner = landmarks[36]  # 左眼的坐标[361,383]
        right_eye_corner = landmarks[45] # 右眼的坐标[488,376]
        image_copy = image.copy()
        # 眼睛绘制两点
        x = [left_eye_corner[0],right_eye_corner[0]]
        y = [left_eye_corner[1],right_eye_corner[1]]
        plt.plot(x, y, color='r')
        plt.scatter(x, y, color='b',s=10)
        for i in range(landmarks.shape[0]):
            pt_position = (int(landmarks[i][0]), int(landmarks[i][1]))
            cv2.circle(image_copy, pt_position, 3, (0, 0, 255), -1)
        plt.imshow(image_copy)
        plt.title('face_eyes')
        plt.show()

        # 得到倾斜角 arctan 对数组中的每一个元素求其反正切值，取值范围为[-pi/2, pi/2]。y/x = tan0 则 0 = actan(y/x)
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0])) # -0.055 左右眼y轴高度差/左右眼x轴距离差
        # 旋转中心为图像中心
        # image size after rotating

        height, width, _ = image.shape # 1029 944
        cos = math.cos(radian) # 0.99
        sin = math.sin(radian) # -0.55

        # 为了包含下旋转后的整张图片，重新规划图片大小。其实也是旋转变换
        new_w = int(width * abs(cos) + height * abs(sin)) # 999
        new_h = int(width * abs(sin) + height * abs(cos)) # 1079

        # translation # 翻译
        Tx = new_w // 2 - width // 2 # 27
        Ty = new_h // 2 - height // 2 # 25
        # 仿射变换，是指在几何中，一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。

        # affine matrix 仿射矩阵
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx], # (2,3)
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])
        # src - 输入图像 , M - 变换矩阵 , dsize - 输出图像的大小, borderValue - （重点！）边界填充值; 默认情况下，它为0。
        # M作为仿射变换矩阵，一般反映平移或旋转的关系，为InputArray类型的2×3的变换矩阵。
        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)# （68，3）
        landmarks_rotate = np.dot(M, landmarks.T).T# (2,3)*(3,68).T->(68,2) #关键点转正
        # rotation angle 旋转的角度
        left_eye_corner = landmarks_rotate[36]  # 左眼的坐标[361,383]
        right_eye_corner = landmarks_rotate[45]  # 右眼的坐标[488,376]
        image_rotate_copy = image_rotate.copy()
        # 绘制两点
        x = [left_eye_corner[0], right_eye_corner[0]]
        y = [left_eye_corner[1], right_eye_corner[1]]
        plt.plot(x, y, color='r')
        plt.scatter(x, y, color='b', s=10)
        for i in range(landmarks_rotate.shape[0]):
            pt_position = (int(landmarks_rotate[i][0]), int(landmarks_rotate[i][1]))
            cv2.circle(image_rotate_copy, pt_position, 3, (0, 0, 255), -1)
        plt.imshow(image_rotate_copy)
        plt.title('image_rotate')
        plt.show()

        return image_rotate, landmarks_rotate # 转正后的图像，转正后的关键点

    def crop(self,image, landmarks,scale):  # (2034,1596,3) (68,2)
        landmarks_top = np.min(landmarks[:, 1])  # 所有行第一列 找出纵坐标最高的点694.2339017087882
        landmarks_bottom = np.max(landmarks[:, 1])  # 找出纵坐标最低的点1588.8271446326971
        landmarks_left = np.min(landmarks[:, 0])  # 找出横坐标最左边的点341.886315439108
        landmarks_right = np.max(landmarks[:, 0])  # 找出横坐标最右边的点1231.441236213454
        # 找出边界
        landmarks_top_index = landmarks[:, 1].tolist().index(landmarks_top)
        landmarks_bottom_index = landmarks[:, 1].tolist().index(landmarks_bottom)
        landmarks_left_index = landmarks[:, 0].tolist().index(landmarks_left)
        landmarks_right_index = landmarks[:, 0].tolist().index(landmarks_right)

        image_copy = image.copy()
        mark_list = [landmarks_top_index, landmarks_bottom_index, landmarks_left_index, landmarks_right_index]
        for i in range(landmarks.shape[0]):
            pt_position = (int(landmarks[i][0]), int(landmarks[i][1]))
            cv2.circle(image_copy, pt_position, 8, (0, 0, 255), -1)
        for index in mark_list:
            pt_position = (int(landmarks[index][0]), int(landmarks[index][1]))
            cv2.circle(image_copy, pt_position, 8, (255, 0, 0), -1)
        plt.imshow(image_copy)
        plt.title('face_top_bottom_left_right')
        plt.show()

        # expand bbox 确定想要截取人脸的大小，想要截取为正方形大小的人脸
        top = int(

            landmarks_top - scale * (landmarks_bottom - landmarks_top))  # -21 bottom-top=894 0.8*894=715 694-715=-21
        bottom = int(landmarks_bottom + 0.3 * (
                    landmarks_bottom - landmarks_top))  # 1857 bottom-top=894 0.3*894=268 1588+268=1857
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))  # 75
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))  # 1498

        if bottom - top > right - left:  # 截取的正方形的高大于宽时，左边扩展（高-宽）的一半，右边扩展（高-宽）的一半，这样截取的就为正方形了
            left -= ((bottom - top) - (right - left)) // 2  # -152
            right = left + (bottom - top)  # 1726 这时left-right=1726+152=1878=bottom-top=1857+21=1878
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)
        # 截取比例放大


        # 确定了剪切图片后的大小，默认为白色
        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255  # （1879，1879，3）

        h, w = image.shape[:2]  # 2034 1596
        left_white = max(0,
                         -left)  # 152 剪切图最左边从哪里开始放，如果-left>0说明要截取的地方大于了原图本身（所以放在截取图的left位置），-left<0说明要截取的地方并没超过原图本身（所以直接放在截取图的0位置）
        left = max(0, left)  # 0 原图最左边从哪里开始截取，这里是边界
        right = min(right, w - 1)  # 1596 原图最右边从哪里开始截取，这里是边界
        right_white = left_white + (right - left)  # 1747 截取后应该放在剪切图哪里
        top_white = max(0, -top)  # 21 剪切图最上边从哪里开始放
        top = max(0, top)  # 0 原图从哪里开始截取
        bottom = min(bottom, h - 1)  # 1857
        bottom_white = top_white + (bottom - top)  # 1878

        # 画出边框
        cv2.rectangle(image_copy, (left + 8, top + 8), (right - 8, bottom - 8), (0, 255, 0), 8)
        plt.imshow(image_copy)
        plt.title('face_rectangle')
        plt.show()

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()
        plt.title('image_crop')
        plt.imshow(image_crop)
        plt.show()
        return image_crop  # (1879,1879,3)