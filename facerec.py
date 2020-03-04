import numpy as np
import cv2
import os
from json_minify import json_minify
import json
import face_recognition
import pickle
import pprint
import matplotlib.pyplot as plt

conf = json.loads(json_minify(open("conf.json").read()))
enc = None

if conf["source_enc_file"] == "none": # Ha még nincs enc file akkor csináljunk.
    enc = {"real": {}, "fake": {}}
    print("Starting to extract frames and embeddings from videos...")
    for tp in ["real", "fake"]:
        videoCount = 0
        allVideosOfThisType = os.listdir(tp)
        countOfAllVideosOfThisType = len(allVideosOfThisType)
        for videoFile in allVideosOfThisType:
            videoCount += 1
            pathToVideoFile = tp + os.sep + videoFile
            enc[tp][videoFile] = []
            frameCount = 0
            cap = cv2.VideoCapture(pathToVideoFile)
            while(True):
                ret, frame = cap.read()

                frameCount += 1

                if frameCount % conf["frame_rate"] == 0:
                    print(f"({tp}) Extracting frame {frameCount} from video {videoCount} / {countOfAllVideosOfThisType}")
                    faceEncoding = face_recognition.face_encodings(frame)

                    if len(faceEncoding) == 1: # Sikerült-e egyáltalán kivenni az embeddinget?
                        enc[tp][videoFile].append(faceEncoding[0])

                if frameCount == 20:
                    break

            cap.release()
            cv2.destroyAllWindows()

    print("Starting to calculate centroids for videos...")

    for tp in enc.keys():
        for videoFile, listOfEncodings in enc[tp].items():
            centroid = 0
            sumOfEncodings = 0
            numberOfEncodings = 0
            for encoding in listOfEncodings:
                sumOfEncodings += encoding
                numberOfEncodings += 1
            try:
                centroid = sumOfEncodings / numberOfEncodings
            except:
                print(f"ERROR! Coudn't compute centroid for {videoFile} (division by zero)")
                print(f"List of encodings: {listOfEncodings}")

            enc[tp][videoFile].append({videoFile + "_centroid": centroid})

    with open(conf["dest_enc_file"], 'wb') as f:
        pickle.dump(enc, f)
else: # Ha már van enc file akkor használjuk azt.
    with open(conf["source_enc_file"], 'rb') as f:
        enc = pickle.load(f)

# Calculate distances from centroid

distancesDict = {"real": {}, "fake": {}}
for tp in enc.keys():
    for videoFile, encodingsAndCentroid in enc[tp].items():
        distancesDict[tp][videoFile] = []
        centroid = encodingsAndCentroid[-1][videoFile + "_centroid"]
        for encoding in encodingsAndCentroid[:-1]:
            distanceFromCentroid = np.linalg.norm(centroid-encoding)
            distancesDict[tp][videoFile].append(distanceFromCentroid)

for tp in distancesDict.keys():
    for videoFile, distances in distancesDict[tp].items():
        try:
            plt.plot(distances, marker=".", linestyle='None')
            ymax = max(distances)*1.2
            plt.ylim(0, ymax)
            plt.savefig(videoFile.split(".")[0] + '.png')
            plt.clf()
        except:
            print(f"WARNING! Coudn't compute ymax for {videoFile} (zero element)")