import cv2
import insightface
import face_recognition
import json
from json_minify import json_minify
import pickle
import pprint

with open("IFencodings.pickle", "rb") as f:
    pprint.pprint(pickle.load(f))


