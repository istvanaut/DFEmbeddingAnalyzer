import numpy as np
import cv2
import os
from json_minify import json_minify
import json
import face_recognition

enc = {"real": {}, "fake": {}}

conf = json.loads(json_minify(open("conf.json").read()))

# TODO: ha konyvtarat adtak meg, az osszes videot vegig kell jarni.
enc["real"][os.path.basename(conf["real"])] = []
enc["fake"][os.path.basename(conf["fake"])] = []

for tp in ["fake", "real"]:
    for fn in enc[tp].keys():
        print(fn)
    #     fix = 0
    #     cap = cv2.VideoCapture(conf[tp]) # TODO: iterate filepaths
    #     while(True):
    #         ret, frame = cap.read()
    #         fix += 1
    #         print(fix)
    #
    #         if fix % conf["frame_rate"] == 0:
    #             enc[tp][fn].append(face_recognition.face_landmarks(frame))
    #
    #         if fix == 30:
    #             break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()