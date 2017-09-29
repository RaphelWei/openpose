#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:35:38 2017

@author: lthpc
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 as cv 
#import numpy as np
#import math
import caffe
import argparse
import time
from config_reader import config_reader
from pose_util import get_paf_heatmap_avg
from pose_util import get_paf_heatmap_avg_single_scale
from pose_util import get_all_peak
from pose_util import get_connection_all_special_k
from pose_util import get_candidate_subset
from pose_util import render
from pose_util import keypoint2track

from timer import Timer
import numpy as np
import alarm_openpose
from alarm_openpose import *
from configobj import ConfigObj

'''
&&&&&&&&&&&&&&&&&&&
main_win root 路径
&&&&&&&&&&&&&&&&&&&
'''
from sys import path
path.append(r'/home/bob/DeepAction+') #将存放module的路径添加进来
from all_path import main_win_root

param, model,config = config_reader()

#if param['use_gpu']: 
MER_RATIO=config['ALARMREGION']
# find connection in the specified sequence, center 29 is in the position 15
#limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
 #          [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
  #         [1,16], [16,18], [3,17], [6,18]]

limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,1]]
# the middle joints heatmap correpondence
#mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
#          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
 #         [55,56], [37,38], [45,46]]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [47,48]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

#pSeq = [1,2,3,4,5,6,7,9,10,12,13]
pSeq = [1,2,3,4,5,6,7]

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Opense camera address')
    parser.add_argument('--cam0', dest='cam0_name', help='camera index 0 address',default='')
    args = parser.parse_args()
    return args

class parameter(object):
        
    def __init__(self, argv):
        #self.gpu_id=0
        #self.net_name="zf"
        #self.sotf_nms=1
        #self.cam0_name="../data/demo/test3_000.mp4"
        #self.cam1_name="../data/demo/test3_003.mp4"
        #self.cam2_name="../data/demo/test3_014.mp4"
        self.cam0_address=argv["cam0_address"]
        self.cam0_id = argv["cam0_id"]
        self.cam0_name = argv["cam0_name"]

class openpose(QtCore.QThread):
    pic_sig = QtCore.pyqtSignal(str, int, int, int, object)
    info_sig = QtCore.pyqtSignal(object)
    warn_sig = QtCore.pyqtSignal(str, str, int, object)
    address_sig = QtCore.pyqtSignal(str)
    def __init__(self,argv=None, alarm_con=None):
        super(openpose, self).__init__()
        self.alarm_con = alarm_con
        #self.alarm_con['hands_up_tag']=True
        #self.alarm_con['fighting_tag']=False
        #self.alarm_con['hands_wave_tag']=False
        #self.alarm_con['hand_alarm_level'] = 0 # 0--(3>2 or 6>5) 1--((3>2 and 4>2) or (6>5 and 7>5) 2--(3>2 and 6>5)
        #self.alarm_con['hand_wave_level'] = 0
        #self.alarm_con['fighting_alarm_level']=0
        #self.alarm_con['ROI_RATIO'] = [0.33,0.05,0.05,0.05]
        
        self.alarm_config = []
        self.alarm_config.append(self.alarm_con)
        self.alarm_config.append(self.alarm_con)

        self.args = parameter(argv)
        self.identity = None
        self.address = [self.args.cam0_name, ""]
        self.id = [self.args.cam0_id, ""]
        self.name = [self.args.cam0_name, None]
        self.main_tag = [0, 0]
        self.type = "openpose"
        self.video_id = [0, 0]
        self.warn_count = 0

    def run(self):
        SNAP_FRAME = 30
        firstframe = True
        keypoints = []
        cnt = 1
        collectpoints = []
        num_object = 0
        zero_count=True #cout for poitlist
        args = self.args # get args
        self.address.append(args.cam0_name)
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber']) # set to your device!
        net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
        
        cam_names=args.cam0_address

        while (1):
           video=cv.VideoCapture(cam_names)
           if (not video.isOpened()):
               print 'Dinsconect with camera,restart after 5 seconds...'
               time.sleep(5)
           else:
               break
        #video=cv.VideoCapture(cam_names)
        if video.isOpened():
            success, frame= video.read()
            while success:
                alarm_info_dict={}
                alarm_info_dict['hands_wave']=[]
                alarm_info_dict['hands_up']=[]
                alarm_info_dict['fighting']=[]
                #img_ori = cv.resize(frame,(frame.shape[1],frame.shape[0]),interpolation=cv.INTER_CUBIC);
                img = cv.resize(frame,(640,360),interpolation=cv.INTER_CUBIC);
                img_ori=img.copy()
                #timer = Timer()
                #timer.tic()
                if firstframe:
                    paf_avg,heatmap_avg=get_paf_heatmap_avg_single_scale(net,img,model,param)
                    all_peaks=get_all_peak(heatmap_avg,param)
                    connection_all,special_k=get_connection_all_special_k(img,all_peaks,paf_avg,param,mapIdx,limbSeq)
                    candidate,subset=get_candidate_subset(mapIdx,limbSeq,all_peaks,connection_all,special_k)
                    keypoints = keypoint2track(subset,candidate,pSeq)
                    collectpoints.append(keypoints.tolist())
                    pre_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    firstframe = False
                else:
                    cur_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
                    num_object = len(keypoints)
                    for i in range(num_object):
                        keypoints[i], st, err = cv.calcOpticalFlowPyrLK(pre_gray, cur_gray, keypoints[i], None, **lk_params)
                    collectpoints.append(keypoints.tolist())
                #alarm.drawTrajectory(img, collectpoints, colors, pSeq)
                drawTrajectory(img, collectpoints, colors, pSeq)
                #keypointangles = alarm.getangle2calcentropy(collectpoints, num_object)
                #keypointvelocitys = alarm.getvelocity2calcenergy(img.shape[0], collectpoints, num_object)
                keypointvelocitys = getvelocity2calcenergy(img.shape[0], collectpoints, num_object)
                num_object=len(keypointvelocitys)
                if zero_count and num_object!=0:
                    count_list=[0]*num_object
                    zero_count=False
                if num_object!=0:
                    NOT2ALARMRATIO=self.alarm_config[0]['ROI_RATIO']
                    MER_RATIO[0]=max(MER_RATIO[0],NOT2ALARMRATIO[0])
                    MER_RATIO[1]=max(MER_RATIO[1],NOT2ALARMRATIO[1])
                    MER_RATIO[2]=max(MER_RATIO[2],NOT2ALARMRATIO[2])
                    MER_RATIO[3]=max(MER_RATIO[3],NOT2ALARMRATIO[3])

                    img, energy,= unite2alarm(img, keypointvelocitys, keypoints, pSeq, cnt, \
                                                    self.alarm_config[0]['hands_up_tag'], \
                                                    self.alarm_config[0]['fighting_tag'],\
                                                    self.alarm_config[0]['hands_wave_tag'], \
                                                    self.alarm_config[0]['hand_alarm_level'], \
                                                    self.alarm_config[0]['fighting_alarm_level'],\
                                                    self.alarm_config[0]['hand_wave_level'], \
                                                    MER_RATIO, \
                                                    count_list,SNAP_FRAME, \
                                                    alarm_info_dict)
                    x1=int(img.shape[1]*NOT2ALARMRATIO[2])
                    y1=int(img.shape[0]*NOT2ALARMRATIO[0])
                    x2=int(img.shape[1]*(1-NOT2ALARMRATIO[3]))
                    y2=int(img.shape[0]*(1-NOT2ALARMRATIO[1]))
                    cv.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
                #img, energy = alarm.unite2alarm(img, keypointvelocitys, keypoints, pSeq, cnt,self.config_alarm['hands_up_tag'],self.config_alarm['fighting_tag'],\
                #                                self.config_alarm['hands_wave_tag'],self.config_alarm['hand_alarm_level'],self.config_alarm['fighting_alarm_level'],\
                #                                self.config_alarm['hand_wave_level'])
                
                cnt += 1
                #timer.toc()
                #cv.putText(img, 'fps: %.2f' % (1/timer.total_time), (0,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)


                
                pre_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
                if cnt%SNAP_FRAME == 0:
                    firstframe=True
                    cnt=1
                    keypoints = []
                    collectpoints = []
                    num_object = 0
                    zero_count=True
                success,frame = video.read()
                #if camera disconect in the process
                if not success:
                    img=cv.imread(main_win_root + 'LossSignal.jpg')
                    firstframe=True
                    cnt=1
                    keypoints = []
                    collectpoints = []
                    num_object = 0
                    zero_count=True
                    while(1):
                        video=cv.VideoCapture(cam_names)
                        if (not video.isOpened()):
                         ###+++ modify here to qt show
                        # for qt show
                        #+++++++++++++++++++++++change here for QT show+++++++++++++++++++++++++++++                           
                            cimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                            height, width, bytesPerComponent = cimg.shape
                            qimage = QtGui.QImage(cimg.tostring(), width, height, QtGui.QImage.Format_RGB888)
                            pic = QtGui.QPixmap.fromImage(qimage)
                            self.pic_sig.emit(self.type, self.identity, 0, self.video_id[0], pic)                            
          
                        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        #+++++++++++++++++++++++++
                            print 'Dinsconect with camera,restart after 5 seconds...'
                            time.sleep(5)
                        else:
                            success, frame= video.read()
                            if success:
                                break
                
                cv.imshow("mat", img)
                cv.waitKey(1)
                # for qt show
                #+++++++++++++++++++++++change here for QT show+++++++++++++++++++++++++++++
                '''
                cimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                height, width, bytesPerComponent = cimg.shape
                qimage = QtGui.QImage(cimg.tostring(), width, height, QtGui.QImage.Format_RGB888)
                pic = QtGui.QPixmap.fromImage(qimage)
                self.pic_sig.emit(self.type, self.identity, 0, self.video_id[0], pic)
                '''
                warnning_type = ['hands_wave', 'hands_up','fighting' ]                
                 
                if alarm_info_dict["hands_wave"] != [] :
                    warn_infos = []
                    warn_infos.append(alarm_info_dict["hands_wave"])
                    warn_infos.append(self.id[0] + self.name[0])
                    warn_infos.append(warnning_type[0])
                    self.info_sig.emit(warn_infos)
                    self.warn_sig.emit(self.id[0] + self.name[0], warnning_type[2], self.warn_count, pic)
                    self.warn_count = self.warn_count + 1
                    self.warn_count = self.warn_count % 4
                
                if alarm_info_dict['hands_up'] != [] :
                    warn_infos = []
                    warn_infos.append(alarm_info_dict['hands_up'])
                    warn_infos.append(self.id[0] + self.name[0])
                    warn_infos.append(warnning_type[1])
                    self.info_sig.emit(warn_infos)
                    self.warn_sig.emit(self.id[0] + self.name[0], warnning_type[2], self.warn_count, pic)
                    self.warn_count = self.warn_count + 1
                    self.warn_count = self.warn_count % 4
                
                if alarm_info_dict['fighting'] != [] :
                    warn_infos = []
                    warn_infos.append(alarm_info_dict['fighting'])
                    warn_infos.append(self.id[0] + self.name[0])
                    warn_infos.append(warnning_type[2])
                    self.info_sig.emit(warn_infos)
                    self.warn_sig.emit(self.id[0] + self.name[0], warnning_type[2], self.warn_count, pic)
                    self.warn_count = self.warn_count + 1
                    self.warn_count = self.warn_count % 4
                
                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #cv.destroyAllWindows()
            #video.release()

if __name__ == '__main__':
    argv = {}
    argv["cam0_address"] = "../sample_image/test3.mp4"
    argv["cam0_id"] = "AS"
    argv["cam0_name"] = "红鲤鱼"

    config = ConfigObj(main_win_root + 'alarm_config.ini', encoding='UTF8')
        #print  config['track_alarm_config'], config['pose_alarm_config']
    pose_alarm_config = {}

    #self.track_alarm_config['ROI_RATIO'] = [float(config['track_alarm_config']['ROI_RATIO'][0].lstrip('[')), 
    #                                        float(config['track_alarm_config']['ROI_RATIO'][1]),
    #                                        float(config['track_alarm_config']['ROI_RATIO'][2]),
    #                                        float(config['track_alarm_config']['ROI_RATIO'][3].rstrip(']')) ] # alram roi ratio : up down left right
    pose_alarm_config['ROI_RATIO'] = [ float(i) for i in config['pose_alarm_config']['ROI_RATIO']]
    #print self.track_alarm_config['ROI_RATIO']          
    pose_alarm_config['hands_up_tag'] = config['pose_alarm_config']['hands_up_tag'] # whether do pedestrian count alarm
    pose_alarm_config['fighting_tag'] = config['pose_alarm_config']['fighting_tag'] # whether do pedestrian count alarm
    pose_alarm_config['hands_wave_tag'] = config['pose_alarm_config']['hands_wave_tag'] # whether do pedestrian count alarm
    pose_alarm_config['hand_alarm_level'] = int(config['pose_alarm_config']['hand_alarm_level']) # whether do pedestrian count alarm
    pose_alarm_config['hand_wave_level'] = int(config['pose_alarm_config']['hand_wave_level']) # whether do pedestrian count alarm
    pose_alarm_config['fighting_alarm_level'] = int(config['pose_alarm_config']['fighting_alarm_level']) # whether do pedestrian count alarm
        
    cat = openpose(argv, pose_alarm_config)
    cat.run()
