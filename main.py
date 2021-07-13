import os
import cv2
import numpy as np
import torch
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
from dataloater.u2net_data_loader import SalObjDataset
from dataloater.u2net_data_loader import RescaleT
from dataloater.u2net_data_loader import ToTensorLab
from torchvision import transforms
from utils.face_detect import FaceDetect
from model.u2net import U2NET
from torch.utils.data import DataLoader
from PIL import Image
from model.pix2pix import UnetGenerator
from model.photo2cartoon import ResnetGenerator as Photo2Cartoon_ResnetGenerator
from model.ugatit import ResnetGenerator as UGATIT_ResnetGenerator
from utils.transform import scaling_image,put_text,blend_images
from utils.face_seg import FaceSeg

def get_face(image,scale,img_name):
    """剪切出人脸"""
    device = 'cpu'
    detector = 'dlib'
    detect = FaceDetect(device, detector)
    face_info = detect.align(image)
    image_align, landmarks_align = face_info  # 旋转后人脸得到矫正的图片(1079,999,3) 人脸关键点的坐标(68,2)
    face = detect.crop(image_align, landmarks_align,scale)  # 剪切旋转后的原图，得到人脸(429,429,3)
    # face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./dataset/face/'+img_name, face[:,:,::-1])
    return face

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):
    # image_name 'D:\\ZXHcode\\Segmentation\\U-2-Net-master\\test_data\\test_human_images_results\\' pred (1,320,320) d_dir 'D:\\ZXHcode\\Segmentation\\U-2-Net-master\\test_data\\test_human_images_results\\'
    predict = pred# (1,320,320)
    predict = predict.squeeze() #320 320
    predict_np = predict.cpu().data.numpy() # (320,320)

    im = Image.fromarray(predict_np*255).convert('RGB')# (320,320)
    img_name = image_name.split(os.sep)[-1] # ao.jpg
    image = cv2.imread(image_name) #(1600,1600,3) 原图
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)#(1600,1600)

    # pb_np = np.array(imo)

    aaa = img_name.split(".") # ao jpg
    bbb = aaa[0:-1]# ao
    imidx = bbb[0] # ao
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')
    imo.save(d_dir)
    imo = cv2.imread(d_dir)
    imo_gray = cv2.cvtColor(imo,cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(imo_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(d_dir,binary)

    plt.title('face_mask')
    plt.imshow(binary)
    plt.show()
    return binary

def get_mask_FCN(face,img_name):
    prediction_dir = os.path.join(os.getcwd(), 'dataset', 'result_seg' , img_name)#'D:\\ZXHcode\\Segmentation\\U-2-Net-master\\test_data\\test_human_images_results\\'
    segment = FaceSeg()
    mask = segment.get_mask(face)
    cv2.imwrite(prediction_dir, mask)
    plt.title('face_mask')
    plt.imshow(mask)
    plt.show()
    return mask

def get_mask_U2net(face,img_name):
    # 模型路径
    model_dir = os.path.join(os.getcwd(), 'save_model', 'u2net_human_seg.pth')
    # 测试路径
    img_name_list = [os.path.join(os.getcwd(), 'dataset','face', img_name),]
    # 预测路径
    prediction_dir = os.path.join(os.getcwd(), 'dataset', 'result_seg' , img_name)

    # Dataset
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    u2net = U2NET(3,1)
    u2net.load_state_dict(torch.load(model_dir,map_location='cpu'))
    u2net = u2net.cuda()
    u2net.eval()

    # 测试
    for i_test, data_test in enumerate(test_salobj_dataloader):  # data_set imidx:(1,1) image(1,3,320,320) label(1,1,320,320)
        inputs_test = data_test['image']  # (1,3,320,320)
        inputs_test = inputs_test.type(torch.FloatTensor)  # (1,3,320,320)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d1, d2, d3, d4, d5, d6, d7 = u2net(inputs_test)  # 7个(1,1,320,320)

        # normalization
        pred = d1[:, 0, :, :]  # (1,320,320)
        pred = normPRED(pred)

        # save results to test_results folder

        pred = save_output(img_name_list[i_test], pred, prediction_dir)
        return pred

def get_face_white_bg_photo2cartoon(face_rgba,img_name):
    face = face_rgba[:, :, :3].copy()  # 429 429 3 #取出前三个通道
    mask = face_rgba[:, :, 3].copy()[:, :, np.newaxis] / 255.  # 429 429 1 取出mask
    face_white_bg = (face * mask + (1 - mask) * 255).astype(np.uint8)  # 429 429 3 背景变白色

    plt.title('face_white_bg')
    plt.imshow(face_white_bg)
    plt.show()
    cv2.imwrite(os.path.join(os.getcwd(), 'dataset','result_white_bg', img_name) ,
                cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))
    # return cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR)

    face = face_rgba[:, :, :3].copy()
    mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
    face_white_bg = (face * mask + (1 - mask) * 255) / 127.5 - 1
    return face_white_bg

def get_face_white_bg_pix(face_rgba,img_name):
    face = face_rgba[:, :, :3].copy()  # 429 429 3 #取出前三个通道
    mask = face_rgba[:, :, 3].copy()[:, :, np.newaxis] / 255.  # 429 429 1 取出mask
    face_white_bg = (face * mask + (1 - mask) * 255).astype(np.uint8)  # 429 429 3 背景变白色


    plt.title('face_white_bg')
    plt.imshow(face_white_bg)
    plt.show()
    cv2.imwrite(os.path.join(os.getcwd(), 'dataset','result_white_bg', img_name) ,
                cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR)


def get_cartoon_face_pix2pix(image,img_name):
    generator = UnetGenerator()
    last_weights_path = os.path.join(os.path.join(os.getcwd(), 'save_model','pix2pix.pth'))
    # print('加载权重:', last_weights_path)
    model_state_dict = torch.load(last_weights_path,map_location=torch.device('cpu'))  # 读取权重
    generator.load_state_dict(model_state_dict)  # 读取权重加载到生成器当中
    generator.eval()  # 生成器调整到测试模式
    img_A = cv2.resize(image, (256, 256))  # 先把真人照给他挑出来
    g_input = img_A.astype('float32') / 127.5 - 1  # 归一化到-1-1
    g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # 把通道数放在前面NHWC -> NCHW
    g_input = torch.from_numpy(g_input)  # 把numpy -> torch可以执行的tensor

    g_output = generator(g_input)  # 把tensor放到生成器当中
    g_output = g_output.detach().numpy()  # tensor -> numpy
    g_output = g_output.transpose(0, 2, 3, 1)[0]  # 把通道放在最后面NCHW -> NHWC，由于batch_size为1，所以[0]把他读取出来
    g_output = g_output * 127.5 + 127.5  # 反归一化到0-255
    g_output = g_output.astype(np.uint8)  # 转化为unit8

    img_show = g_output[:, :, ::-1]  # 把BGR的图片转化为RGB， ::-1本来为通道，因为opencv读的图是BGR的，但是plt支持RGB

    img_show = cv2.resize(img_show,(512,512))
    plt.title('cartoon')
    plt.imshow(img_show)
    plt.show()

    cv2.imwrite(os.path.join(os.getcwd(), 'dataset','cartoon_face', img_name) ,
                cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
    return img_show

def get_cartoon_face_photo2cartoon(face_white_bg,mask,img_name):
    mask = cv2.resize(mask, (256, 256))
    mask =  mask[:, :, np.newaxis].copy() / 255.


    face_white_bg = cv2.resize(face_white_bg, (256, 256), interpolation=cv2.INTER_AREA)
    face_white_bg = np.transpose(face_white_bg[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
    face_white_bg = torch.from_numpy(face_white_bg)

    model_dir = os.path.join(os.getcwd(), 'save_model', 'photo2cartoon_weights.pt')

    net = Photo2Cartoon_ResnetGenerator(ngf=32, img_size=256, light=True)
    params = torch.load(model_dir,map_location='cpu')
    net.load_state_dict(params['genA2B'])
    cartoon = net(face_white_bg)[0][0]
    # post-process
    cartoon = np.transpose(cartoon.cpu().detach().numpy(), (1, 2, 0))
    cartoon = (cartoon + 1) * 127.5
    cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    cartoon = cv2.resize(cartoon,(512,512))[:,:,::-1]

    plt.imshow(cartoon)
    plt.title('cartoon')
    plt.show()

    cv2.imwrite(os.path.join(os.getcwd(), 'dataset', 'cartoon_face', img_name)
               ,cartoon[:,:,::-1])
    return cartoon

def get_cartoon_face_ugatit(face_white_bg, mask, img_name):
    mask = cv2.resize(mask, (256, 256))
    mask = mask[:, :, np.newaxis].copy() / 255.

    face_white_bg = cv2.resize(face_white_bg, (256, 256), interpolation=cv2.INTER_AREA)
    face_white_bg = np.transpose(face_white_bg[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
    face_white_bg = torch.from_numpy(face_white_bg)

    params = torch.load('./save_model/u_gat_it.pt')
    genA2B = UGATIT_ResnetGenerator(input_nc=3, output_nc=3, ngf=32, n_blocks=4, img_size=256,
                                  light=True)
    genA2B.load_state_dict(params['genA2B'])
    cartoon, _, _ = genA2B(face_white_bg)

    cartoon = np.transpose(cartoon[0].cpu().detach().numpy(), (1, 2, 0))
    cartoon = (cartoon + 1) * 127.5
    cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    cartoon = cv2.resize(cartoon, (512, 512))[:, :, ::-1]

    plt.imshow(cartoon)
    plt.title('cartoon')
    plt.show()

    cv2.imwrite(os.path.join(os.getcwd(), 'dataset', 'cartoon_face', img_name)
                , cartoon[:, :, ::-1])
    return cartoon


def merge_process(im, bg,mask,output_path):#
    h, w = im.shape[:2] # 128 128
    bh, bw = bg.shape[:2] # 512 512

    wratio = w / bw  # 0.25
    hratio = h / bh  # 0.25

    ratio = wratio if wratio > hratio else hratio  # 1
    # Math.ceil()  向上取整
    if ratio > 1: # 原图比背景图大
        bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)

    im = np.array(im, np.float32)[:,:,::-1] # 128 128
    # im = np.array(bg, np.float32)[:, :, ::-1]
    bg_h, bg_w = bg.shape[:2]  # 512 512
    x = max(0, int((bg_w - w) / 2))  # 192
    y = max(0, int((bg_h - h) / 2))  # 192
    # crop = np.array(bg[bh- h:bh , 0:x + w]) # 截取背景与原图同样大小的地方(128,128,3)
    #
    # plt.imshow(crop)
    # plt.title('crop')
    # plt.show()

    crop = np.array(bg[bh- h:bh, 0+y: w+y], np.float32) # 截取背景与原图同样大小的地方(512,512,3)

    alpha = np.zeros((h, w, 1), np.float32)  # （128,128，1） 全为0
    alpha[:, :, 0] = mask / 255.  # (128,128,1) # 分割后的图像给alpha

    im = alpha * im + (1 - alpha) * crop #  （973,808,3）分割后的图像*原图 + （1-分割后的图像）*背景图
    bg[bh- h:bh ,0+y: w+y] = im

    cv2.imwrite(output_path, bg) # 荣和
    plt.imshow((bg/255)[:,:,::-1])
    plt.title('merge')
    plt.show()

    return bg

def parse_opt():
    parser = argparse.ArgumentParser(description='Photo2Cartoon')

    # 人像图片名字
    parser.add_argument('--img-name',type=str,default='nini.png',help='Image name')
    # 融合图片名字
    parser.add_argument('--background-name',type=str,default='yourname2.jpeg',help='Background image name')
    # 写在图片上的字
    parser.add_argument('--text-content',type=str,default='nini',help='The words written on the cartoon picture')
    # 字的大小
    parser.add_argument('--text-scale',type=int,default=70,help='The words size')
    # 字的位置
    parser.add_argument('--text-location',type=tuple,default=(220,30),help='The words location')
    # 融合方式，是前景融合还是背景融合
    parser.add_argument('--fusion-method',type=str,default='pre_fusion',help='[pre_fusion/back_fusion]')
    # 头部剪切比例，值越大剪切比例也大
    parser.add_argument('--shear-rate',type=int,default=0.8,help='Head cut rate')
    # 选择分割模型，有FCN和U2net，经本人测试两种各有优劣，没有说谁一定更好
    parser.add_argument('--segment-model',type=str,default='Unet',help='[FCN/U2net]')
    # 卡通风格迁移方法，强烈建议用Photo2cartoon模型，其他两个模型有时迁移效果很阴间
    parser.add_argument('--migration-method',type=str,default='Photo2cartoon',help='[Photo2cartoon/U-GAT-IT/Pix2pix]')

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":

    opt = parse_opt()
    # 读取图片,并且显示

    img = cv2.cvtColor(cv2.imread(os.path.join('dataset','img', opt.img_name)), cv2.COLOR_BGR2RGB)
    plt.title('img')
    plt.imshow(img)
    plt.show()

    # 识别人脸关键点->人脸摆正->分割人脸
    # 参数： 原图，裁剪比例，图片名字
    face = get_face(img,opt.shear_rate,opt.img_name)

    # 分割图像
    # 得到切割后的黑白图
    if opt.segment_model == 'U2net':
        mask = get_mask_U2net(face,opt.img_name)
    else:
        mask = get_mask_FCN(face,opt.img_name)

    # 通过分割图像将人像的背景变为白色
    # 使用photo2cartoon模型进行卡通风格迁移
    if opt.migration_method == 'Photo2cartoon':
        face_white_bg = get_face_white_bg_photo2cartoon(np.dstack((face, mask)),opt.img_name) # 分割后的人脸
        cartoon_face = get_cartoon_face_photo2cartoon(face_white_bg,mask,opt.img_name) # 卡通图
    elif opt.migration_method == 'U-GAT-IT':# 使用U-GAT-IT做风格迁移
        face_white_bg = get_face_white_bg_photo2cartoon(np.dstack((face, mask)),opt.img_name) # 分割后的人脸
        cartoon_face = get_cartoon_face_ugatit(face_white_bg,mask,opt.img_name)
    else:# 使用pix2pix模型进行卡通风格迁移
        face_white_bg = get_face_white_bg_pix(np.dstack((face, mask)),opt.img_name)
        cartoon_face = get_cartoon_face_pix2pix(face_white_bg,opt.img_name)

    # 前景融合
    if opt.fusion_method == 'pre_fusion':
        # 读取图片
        background_img = cv2.imread(os.path.join(os.getcwd(), 'dataset', 'back_ground', opt.background_name))
        # 前景融合后保存路径
        output_path = 'dataset/pre_fuse_output/' + opt.img_name[:-4] +'-'+ opt.background_name[:-4]+'_cartoon.jpg'
        # 对背景进行缩放，缩放到与卡通图像一样的大小
        background = scaling_image(background_img,opt.background_name)
        background_text = put_text('background_resize',opt.background_name ,opt.text_content,opt.text_scale,opt.text_location) # 加上文字
        blend_images(cartoon_face,background_text,output_path) #漫画与背景融合
    else:# 背景融合
        # 背景融合后保存路径
        output_path = 'dataset/back_fuse_output/' + opt.img_name[:-4] +'-'+ opt.background_name[:-4]+'_cartoon.jpg'
        # 读取图片
        background_img = cv2.imread(os.path.join(os.getcwd(), 'dataset', 'back_ground', opt.background_name),cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img,(512,512))
        cv2.imwrite(os.path.join(os.getcwd(), 'dataset', 'background_resize', opt.background_name),background_img)
        background_text = put_text('background_resize',opt.background_name,opt.text_content,opt.text_scale,opt.text_location)  # 加上文字

        background_text = cv2.imread(os.path.join(os.getcwd(), 'dataset', 'background_text', opt.background_name),cv2.COLOR_BGR2RGB)
        cartoon_face = cv2.resize(cartoon_face, (384, 384))
        mask = cv2.resize(mask, (384, 384))
        plt.title('background_img')
        plt.imshow(background_img[:,:,::-1])
        plt.show()

        merge_cartoon =merge_process(cartoon_face,background_text,mask,output_path)

