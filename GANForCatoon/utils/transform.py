import matplotlib.pyplot as plt
from PIL import Image

import os
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import cv2


# 旋转
def rotate_images(original_rotate_img, rotate_img, angle):
    # 读取图片
    # original_rotate_img = Image.open(original_rotate_img)
    original_rotate_img = cv2.imread(original_rotate_img)
    h,w = original_rotate_img.shape[:2]
    # 将图片旋转45度
    # rotate_image = original_rotate_img.rotate(angle)
    M = cv2.getRotationMatrix2D((h/2,w/2),angle,0.8)
    rotate_image = cv2.warpAffine(original_rotate_img,M,(h,w), borderValue=(255, 255, 255))[:,:,::-1]
    # 保存图片
    cv2.imwrite(rotate_img,rotate_image[:,:,::-1])

    # 显示旋转后的图片
    plt.imshow(rotate_image)
    plt.title('rotate_image')
    plt.show()

#镜像
def lr_inversion_image(original_mirror_img,lr_mirror_img):
    #读取图片
    # original_mirror_img = Image.open(original_mirror_img)
    original_mirror_img = cv2.imread(original_mirror_img,cv2.COLOR_BGR2RGB)
    #左右镜像
    # mirror_image = original_mirror_img.transpose(Image.FLIP_LEFT_RIGHT)
    mirror_image = cv2.flip(original_mirror_img,1)
    #保存图片
    cv2.imwrite(lr_mirror_img,mirror_image)

    #展示左右镜像图片
    plt.imshow(mirror_image[:,:,::-1])
    plt.title('lr_mirror_image')
    plt.show()

def bt_inversion_image(original_mirror_img,bt_mirror_img):
    #打开图片
    #original_mirror_img = Image.open(original_mirror_img)
    original_mirror_img = cv2.imread(original_mirror_img, cv2.COLOR_BGR2RGB)
    #上下镜像
    mirror_image = cv2.flip(original_mirror_img,0)

    #保存图片
    cv2.imwrite(bt_mirror_img,mirror_image)

    #展示上下镜像图片
    plt.imshow(mirror_image[:, :, ::-1])
    plt.title('lr_mirror_image')
    plt.show()

#剪切
def shear_image(original_shear_img,shear_img,crop):
    #打开图片
    # original_shear_img = Image.open(original_shear_img)
    original_shear_img = cv2.imread(original_shear_img, cv2.COLOR_BGR2RGB)
    h,w = original_shear_img.shape[:2]
    #剪切
    white_bg = np.zeros(crop,dtype=np.uint8)+ 255
    # print(crop[0]-h)
    # print(crop[1]-w)
    white_bg[crop[0]-h:crop[0],crop[1]-w:crop[1],:] = original_shear_img
    # print(crop_result.size)

    #保存图片
    cv2.imwrite(shear_img,white_bg)

    #展示图片
    plt.imshow(white_bg[:, :, ::-1])
    plt.title('crop_result')
    plt.show()

#缩放
def scaling_image(background_img,background_name,w=512,h=512):

    #打开图片
    # original_scaling_img = Image.open(original_scaling_img)
    # print(os.path.join(os.getcwd(),data_path,background_name))

    #缩放
    scaling_image = cv2.resize(background_img,(w,h))

    #保存图片
    # scaling_image.save(scaling_img)
    cv2.imwrite(os.path.join(os.getcwd(),'dataset','background_resize',background_name), scaling_image)
    #展示图片
    plt.imshow(scaling_image[:, :, ::-1]) # opencv的接口使用BGR，而matplotlib.pyplot 则是RGB模式
    plt.title('scaling_image')
    plt.show()

    return scaling_image[:, :, ::-1]



def put_text(background,background_name,text,scal=20, loc=(220, 40)):
    # original_scaling_img = cv2.imread(original_scaling_img_back, cv2.COLOR_BGR2RGB)
    # # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    # imgzi = cv2ImgAddText(original_scaling_img, text, (50, 300), font, 1.2, (255, 255, 255), 2)

    # cv2.imwrite(original_scaling_img_back, imgzi)
    # #展示图片
    # plt.imshow(imgzi[:, :, ::-1])
    # plt.title('imgzi')
    # plt.show()
    img_path = os.path.join(os.getcwd(),'dataset',background,background_name)
    save_path =  os.path.join(os.getcwd(),'dataset','background_text',background_name)
    backeground_img = Image.open(img_path)
    draw = ImageDraw.Draw(backeground_img)
    fnt = ImageFont.truetype('STXINGKA.TTF', size=scal)
    draw.text(loc, text, fill='white', font=fnt)

    backeground_img.save(save_path)

    #展示图片
    plt.imshow(backeground_img)
    plt.title('backeground_img_text')
    plt.show()

    return backeground_img

# 合成
def blend_images(fore_image, base_image, output_path):
    # 读入图片
    # base_image = Image.open(base_image).convert('RGB')
    # fore_image = Image.open(fore_image).resize(base_image.size)
    # base_image = cv2.cvtColor(base_image,cv2.COLOR_BGR2RGB)
    # 图片加权合成
    scope_map = np.array(fore_image)[:, :, -1] / 255 #(512,512) 0-1 只要最后一个通道
    scope_map = scope_map[:, :, np.newaxis] # (512,512,1)
    scope_map = np.repeat(scope_map, repeats=3, axis=2) # (512,512,3) # 复制三份，沿着C轴
    # plt.imshow(scope_map)
    # plt.title('scope_map')
    # plt.show()

    # 数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    multiply_img_fore = np.multiply(scope_map, np.array(fore_image)[:, :, :3])
    # plt.imshow(multiply_img_fore)
    # plt.title('multiply_img_fore')
    # plt.show()

    multiply_img_base = np.multiply((1 - scope_map),  np.array(base_image))
    # plt.imshow(multiply_img_base)
    # plt.title('multiply_img_base')
    # plt.show()

    res_image = multiply_img_fore + multiply_img_base
    # 保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(output_path)

    # 展示图片
    plt.imshow(res_image)
    plt.title('res_image')
    plt.show()