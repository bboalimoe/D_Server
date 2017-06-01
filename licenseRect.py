#!/usr/bin/python
#coding:utf-8

import numpy as np
import os
import cv2
import sys
import shutil
import codecs

import configPath
from args import args

from other import draw_boxes
from text_dtc.textDetectionCTPN import textDetection
from char_seg.CharSegmentation import CharSegmentation
from char_reg.TextRecognition import TextRecognition
from char_reg.BuildDictionary import buildDictionary
from text_item.arrangeItems import arrangeItems
#
# textline detector
text_detector = textDetection(configPath.ROOT_PATH, args.gpu_cpu, args.gpu_no, \
                              args.license_size[0], args.license_size[1])
# character segmentor
seg_model = configPath.ROOT_PATH + args.model_set_dir + args.char_seg_model
char_segmentor = CharSegmentation(args.textline_height, args.textline_width,\
                                  seg_model, args.gpu_cpu, args.gpu_no)
# text recognizer
text_recognizer = TextRecognition(configPath.ROOT_PATH, args)
# dic
chn_file_path = configPath.ROOT_PATH + args.model_set_dir + args.char_chn_code
eng_file_path = configPath.ROOT_PATH + args.model_set_dir + args.char_eng_code
char2code, code2char = buildDictionary(chn_file_path, eng_file_path)
# aggrange items
itemizer = arrangeItems(args, configPath)
# debug flag
debug = True
# parameters
textline_std_height = 65
textline_extend = 20
"""
image: image path, absolute path
rect: license position (x0,y0, x1,y1)
status: license version
"""
def process(image=None, rect=None, status=None):
   # initialize
    ret_status = {'0':1.0,\
                  '1':0.0,'2':0.0,'3':0.0,'4':0.0,'5':0.0,\
                  '6':0.0,'7':1.0,'8':0.0,'9':0.0,'10':0.0,'11':0.0,\
                  '12':0.0,'13':0.0}
    ret_info = {'0':'中华人民共和国机动车行驶证'.decode('utf-8'),\
                '1':'','2':'','3':'','4':'','5':'',\
                '6':'','7':'印章'.decode('utf-8'),'8':'','9':'','10':'','11':''}
    # process
    if os.path.isfile(image):
        image_name = os.path.basename(image)
        # read image
        img = cv2.imread(image)
        if img is None:
            print('Image doese not exist!')
            ret_status['12'] = 0
            return ret_status, ret_info
        if rect is None:
            license = img
        else:
            license = img[int(rect[1]):int(rect[3]),int(rect[0]):int(rect[2])]
        # detect textline
        textlines, truelines, license = text_detector.detecText(license)
        if textlines is None or textlines.shape[0] == 0:
            print('Textline does not exist!')
            ret_status['12'] = 1
            return ret_status, ret_info
        if debug:
            im_with_text_lines = draw_boxes(license, textlines, is_display=False, \
                                            caption='textline', wait=True)
            cv2.imwrite('./debug/license_detect/'+ image_name, im_with_text_lines)
        # tilt correct and textline crop, and resize
        textline_imges, textline_poses = [], []
        for i in xrange(textlines.shape[0]):
            textline_name = image_name[0:image_name.find('.')] + '_' + str(i) + '.png'
            textline = textlines[i,:]
            trueline = truelines[i,:]
            pts1 = np.float32([[trueline[0],trueline[1]],[trueline[3],trueline[4]],\
                               [trueline[0],trueline[2]],[trueline[3],trueline[5]]])
            pts2 = np.float32([[textline[0],textline[1]],[textline[2],textline[1]],\
                               [textline[0],textline[3]],[textline[2],textline[3]]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(license, M, (license.shape[1],license.shape[0]))
            textline[0] = max(0, textline[0]-textline_extend)
            textline[2] = min(textline[2]+textline_extend,license.shape[1])
            textline_img = dst[int(textline[1]):int(textline[3]),int(textline[0]):int(textline[2]),:]
            scale = (1.0*args.textline_height)/(1.0*textline_img.shape[0])
            textline_img = cv2.resize(textline_img, None, fx=scale, fy=scale)
            textline_imges.append(textline_img)
            textline_poses.append(textline)
            if debug:
                cv2.imwrite('./debug/textline_crop/'+textline_name, textline_img)
        # filter the uncorrect textlines
        for i in reversed(xrange(len(textline_poses))):
            if textline_poses[i][3]-textline_poses[i][1] > textline_std_height:
                del textline_imges[i]
                del textline_poses[i]
        rect_x0, rect_y0, rect_x1, rect_y1 = 0, 0, 0, 0
        for i in xrange(len(textline_poses)):
            if i == 0:
                rect_x0 = textline_poses[i][0]
                rect_y0 = textline_poses[i][1]
                rect_x1 = textline_poses[i][2]
                rect_y1 = textline_poses[i][3]
                continue
            if textline_poses[i][0] < rect_x0:
                rect_x0 = textline_poses[i][0]
            if textline_poses[i][1] < rect_y0:
                rect_y0 = textline_poses[i][1]
            if textline_poses[i][2] > rect_x1:
                rect_x1 = textline_poses[i][2]
            if textline_poses[i][3] > rect_y1:
                rect_y1 = textline_poses[i][3]
        if debug:
            license_image=license.copy()
            for i in xrange(len(textline_poses)):
                cv2.rectangle(license_image,(textline_poses[i][0],textline_poses[i][1]),\
                             (textline_poses[i][2],textline_poses[i][3]),(0,0,255))
            cv2.rectangle(license_image,(rect_x0,rect_y0),(rect_x1,rect_y1),(255,0,0))
            if len(textline_poses)>=8:
                step = (rect_y1-rect_y0)/8.0
                for i in xrange(8):
                    cv2.rectangle(license_image,(int(rect_x0),int(rect_y0+step*i)),\
                                  (int(rect_x1),int(rect_y0+step*(i+1))),(0,255,0))
            cv2.imwrite('./debug/textline_bbox/'+image_name,license_image)
        # recogntion
        textline_chars, textchar_probs, textchar_charnum, textchar_poses = [],[],[],[]
        for i in xrange(len(textline_imges)):
            textline_name = image_name[0:image_name.find('.')] + '_' + str(i) + '.png'
            bgr_img1 = textline_imges[i].copy()
            rgb_img = textline_imges[i].copy()
            rgb_img[:,:,0] = bgr_img1[:,:,2]
            rgb_img[:,:,1] = bgr_img1[:,:,1]
            rgb_img[:,:,2] = bgr_img1[:,:,0]
            # segmenation
            p1, p2, p3, p4, p5, p6 = char_segmentor.segment(rgb_img)
            p1_int = p1.astype(int)
            p2_int = p2.astype(int)
            p3_int = p3.astype(int)
            p4_int = p4.astype(int)
            p5_int = p5.astype(int)
            p6_int = p6.astype(int)
            if debug:
                bgr_img2 = bgr_img1.copy()
                for j in xrange(p1_int.shape[0]):
                    cv2.rectangle(bgr_img2,(p1_int[j,0],p1_int[j,2]),\
                                  (p1_int[j,1],p1_int[j,3]),(0,0,255))
                cv2.imwrite('./debug/textline_seg1/'+textline_name,bgr_img2)
            if debug:
                bgr_img3 = bgr_img1.copy()
                for j in xrange(p4_int.shape[0]):
                    cv2.rectangle(bgr_img3,(p4_int[j,0],p4_int[j,2]),\
                                  (p4_int[j,1],p4_int[j,3]),(0,255,0))
                cv2.imwrite('./debug/textline_seg2/'+textline_name,bgr_img3)
            # recognition
            chars, probs, charnum, poses = text_recognizer.\
                    recogtext(bgr_img1, p1_int, p2_int, p3_int, p4_int, p5_int, p6_int)
            if debug:
                pre_str = ''
                bgr_img4 = bgr_img1.copy()
                for k in xrange(chars.shape[0]):
                    pre_char = code2char[str(int(chars[k,0]))].encode('utf-8')
                    if charnum[k] == 1:
                        color = (0,0,255)
                    elif charnum[k] == 2:
                        color = (0,255,0)
                    elif charnum[k] == 3:
                        color = (255,0,0)
                    elif charnum[k] == 4:
                        color = (0,255,255)
                    elif charnum[k] == 5:
                        color = (255,0,255)
                    elif charnum[k] == 6:
                        color = (255, 255, 0)
                    cv2.rectangle(bgr_img4,(poses[k,0],poses[k,2]),(poses[k,1],poses[k,3]),color)
                    pre_str = pre_str+pre_char
                cv2.imwrite('./debug/textline_rec/'+textline_name[0:textline_name.find('.')]\
                            +'_'+pre_str+'.png',bgr_img4)
            # 
            textline_chars.append(chars)
            textchar_probs.append(probs)
            textchar_charnum.append(charnum)
            textchar_poses.append(poses)
        # todo: consider use a language model to finetune the text
        textline_text = []
        for i in xrange(len(textline_chars)):
            text = ''
            for j in xrange(textline_chars[i].shape[0]):
                text = text + code2char[str(int(textline_chars[i][j,0]))]
            #print text
            textline_text.append(text)
        # arrange item using heuristic rules
        ret_status, ret_info = itemizer.arrange(textline_poses, textline_text, textline_chars,\
                                                textchar_poses, code2char, ret_status, ret_info,\
                                                (rect_x0,rect_y0,rect_x1,rect_y1))

    else:
        print('Image file does not exist!')
        ret_status['12'] = 0
    count = 0
    for (k,v) in ret_info.items():
        if v != '':
            ret_status[k] = 1.0
            count += 1
    ret_status['13'] = count
    if debug:
        log_file = codecs.open('./debug/log.txt','a','utf-8')
        log_file.write(image+'\n')
        for (k,v) in ret_info.items():
            string = k+':'+v+'\n'
            #print(string)
            log_file.write(string)
        log_file.close()
    return ret_status, ret_info

if __name__ == '__main__':
    if debug:
        shutil.rmtree('./debug')
        os.makedirs('./debug/license_detect')
        os.makedirs('./debug/textline_bbox')
        os.makedirs('./debug/textline_crop')
        os.makedirs('./debug/textline_rec')
        os.makedirs('./debug/textline_seg1')
        os.makedirs('./debug/textline_seg2')
 
    img_dir = configPath.ROOT_PATH + args.raw_image_dir + args.license_image_train
    textline_dir = configPath.ROOT_PATH + args.raw_image_dir + args.textline_image_train
    demo_imnames = os.listdir(img_dir)
    
    count = 0
    for img_name in demo_imnames:
        count = count + 1
        if count < 10000:
            continue
        if count == 10050:
            break
        print('count:'+str(count)+' '+img_name)
        img_path = img_dir + '/' + img_name
        print("img path"+" " + img_path )
        status, info = process(img_path)
        for (k,v) in info.items():
            print(k+':'+v.encode('utf-8'))
 
 
