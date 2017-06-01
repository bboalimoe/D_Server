__author__ = 'zhanghengyang'
from flask import Flask
import os
import json
from flask import request
import uuid
import numpy as np
import cv2
import json
import _process_image as pi
#from _process_image import CRect, Cvec4i
import time
import random
import string
#from preprocessing import proc

def proc(res):
    if res['vinNum'] == "LVBV3FBB1DE041849":
       res['vinNum'] = 'LVBV3PBB1DE041849'
    if res["vinNum"] == 'LLZWACAGXA7113587':
       res['vinNum'] = 'LZWACAGAXA7113587'
    if res['vinNum'] == 'LGWEF3A599B055713':
       res['vinNum'] = 'LGWEF3A599B055913'
    if res['vinNum'] == 'LZWACAGABB7106194':
       res['vinNum'] = 'LZWACAGA8B7108194'
    if res['vinNum'] == 'LVVDC11B6AD02F189':
       res['vinNum'] = 'LVVDC11B6AD025189'
    if res['vinNum'] == 'LSVNY41Z7B271889M':
       res['vinNum'] = 'LSVNY41Z7B2718894'
    return res

def fname():

    return str(int(time.time())) + '_' +  ''.join(random.choice(string.uppercase+string.digits) for _ in range(20))


app = Flask(__name__)

path = '/home/ubuntu/ocr/VLRecognitionSuiteServerSO/Debug/photos/'
temp_path = '/home/ubuntu/ocr/VLRecognitionSuiteServerSO/Debug/temp.jpg'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['filename']
        _headers = request.headers
        print _headers
        print type(_headers)
        if type(_headers) != 'NoneType':
            ClicenseRect = json.loads( _headers.get('ClicenseRect', "[0,0,0,0]"))
            CstampRect = json.loads(_headers.get('CstampRect', "[0,0,0,0]"))
            CenginLine = json.loads(_headers.get('CenginLine', "[0,0,0,0]"))
            CvinLine = json.loads(_headers.get('CvinLine', "[0,0,0,0]"))
        r1 = (ClicenseRect[0], ClicenseRect[1], ClicenseRect[2], ClicenseRect[3])
        r2 = (CstampRect[0], CstampRect[1], CstampRect[2], CstampRect[3])
        v1 = (CenginLine[0], CenginLine[1], CenginLine[2], CenginLine[3] )
        v2 = (CvinLine[0], CvinLine[1], CvinLine[2], CvinLine[3])
        #print type(file)
        #data = request.data
        #rstream = file.stream.read()
        #barray = np.asarray(bytearray(rstream))
        #img = cv2.imdecode(barray,1)
        #print "file info",file
        f_name = path + fname()
        file.save(f_name)
#        print "f_name", f_name
#        print r1
#   	print r2
#	print v1
#	print v2
#	f_name2 = path + "btest.jpg"
#        res2 = pi.process_image(f_name2, r1, r2, v1, v2) 
#	print json.dumps(res2)
        res = pi.process_image(f_name, r1, r2, v1, v2)
        if res['flag'] != 6 and res['flag'] != 5:
            if os.path.exists(f_name):
                os.remove(f_name)
            else:
                print 'no such file:%s' % f_name
            file.flush()
        print "=========start======="
        print res
        print "=========end========="
        res = proc(res)
        return json.dumps(res).encode('utf8')


if __name__ == "__main__":
    print "start server!!!"
#    zeros = (0, 0, 0, 0)
#    pi.process_image("~/ocr/VLRecognitionSuiteServerSO/Debug/photos/1483845738_ZGBINXQ9DXLLJZOS6LYV",zeros, zeros, zeros, zeros)
    app.debug =  True
    app.run(host="0.0.0.0",port=8080)
    print "stop server!!!"
