import numpy as np
import cv2
import os
from json_minify import json_minify
import json
import face_recognition
import pickle
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import time

conf = json.loads(json_minify(open("conf.json").read()))
enc = None
# agesDict = {"real": {}, "fake": {}}
# genderDict = {"real": {}, "fake": {}}

USE_INSIGHTFACE = bool(conf["USE_INSIGHTFACE"])
FRAME_LIMIT = 270 # Break loop once processed this frame from the video
VIDEO_LIMIT = None # Only process this many videos from a folder. (enter None for ALL videos)

if USE_INSIGHTFACE:
    import insightface
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id = -1, nms=0.4)

if conf["source_enc_file"] == "none": # Ha még nincs feature file akkor csináljunk.
    enc = {"real": {}, "fake": {}}
    print("Starting to extract frames and embeddings from videos...")
    for tp in ["real", "fake"]:
        videoCount = 0

        if VIDEO_LIMIT is not None: # Process only a limited number of videos
            allVideosOfThisType = sorted(os.listdir(tp))[0:VIDEO_LIMIT]
        else: # Process all videos in the folder
            allVideosOfThisType = sorted(os.listdir(tp))

        countOfAllVideosOfThisType = len(allVideosOfThisType)
        for videoFile in allVideosOfThisType:
            videoCount += 1
            pathToVideoFile = tp + os.sep + videoFile

            enc[tp][videoFile] = []
            # agesDict[tp][videoFile] = []
            # genderDict[tp][videoFile] = []

            frameCount = 0
            cap = cv2.VideoCapture(pathToVideoFile)
            while(True):
                ret, frame = cap.read()

                frameCount += 1

                if frameCount % conf["frame_rate"] == 0:
                    print(f"({tp}) Extracting frame {frameCount} / {FRAME_LIMIT} from video {videoCount} / {countOfAllVideosOfThisType}"
                          f" ({videoFile})")

                    if USE_INSIGHTFACE:
                        faceEncoding = model.get(frame)

                        if len(faceEncoding) == 1:  # Sikerült-e egyáltalán kivenni az embeddinget?
                            enc[tp][videoFile].append(faceEncoding[0].embedding)
                            # agesDict[tp][videoFile].append(faceEncoding[0].age)
                            # genderDict[tp][videoFile].append(faceEncoding[0].gender)
                        else:
                            enc[tp][videoFile].append(None)
                            # agesDict[tp][videoFile].append(-3)
                            # genderDict[tp][videoFile].append(None)
                            print(f"\nWARNING: Couldn't find embedding for frame {frameCount} for {videoFile}\n")

                    else: # Using dlib
                        faceEncoding = face_recognition.face_encodings(frame)

                        if len(faceEncoding) == 1:  # Sikerült-e egyáltalán kivenni az embeddinget?
                            enc[tp][videoFile].append(faceEncoding[0])
                        else:
                            enc[tp][videoFile].append(None)
                            print(f"\nWARNING: Couldn't find embedding for frame {frameCount} for {videoFile}\n")

                if frameCount == FRAME_LIMIT:
                    break

            cap.release()
            cv2.destroyAllWindows()

    print("\nStarting to calculate centroids for videos...\n")

    countOfRealPlusFakeVideos = len(enc["real"].keys()) + len(enc["fake"].keys())

    videoCount = 0
    for tp in enc.keys():
        for videoFile, listOfEncodings in enc[tp].items():
            videoCount += 1
            print(f"Calculating centroid for video {videoCount} out of {countOfRealPlusFakeVideos}")
            centroid = 0
            sumOfEncodings = 0
            numberOfEncodings = 0
            for encoding in listOfEncodings:
                if encoding is not None:
                    sumOfEncodings += encoding
                    numberOfEncodings += 1
            try:
                if numberOfEncodings != 0:
                    centroid = sumOfEncodings / numberOfEncodings
                else:
                    centroid = None
            except:
                print(f"\nERROR! Coudn't compute centroid for {videoFile} (division by zero)")
                print(f"List of encodings: {listOfEncodings}\n")

            enc[tp][videoFile].append({videoFile + "_centroid": centroid})

    with open(conf["dest_enc_file"], 'wb') as f:
        pickle.dump(enc, f)

else: # Ha már van encoding file akkor használjuk azt.
    with open(conf["source_enc_file"], 'rb') as f:
        enc = pickle.load(f)

print("\nStarting to calculate distances from centroid...\n")

if conf["source_feature_file"] == "none": # Ha még nincs feature file akkor csináljunk.
    countOfRealPlusFakeVideos = len(enc["real"].keys()) + len(enc["fake"].keys())
    distancesDict = {"real": {}, "fake": {}}
    videoCount = 0
    for tp in enc.keys():
        for videoFile, encodingsAndCentroid in enc[tp].items():
            videoCount += 1
            print(f"Calculating and normalizing distances from centroid for video {videoCount} out of {countOfRealPlusFakeVideos}")
            centroid = encodingsAndCentroid[-1][videoFile + "_centroid"]

            if centroid is not None:
                distancesDict[tp][videoFile] = []

                for encoding in encodingsAndCentroid[:-1]:
                    if encoding is not None:
                        distanceFromCentroid = np.linalg.norm(centroid - encoding)
                        distancesDict[tp][videoFile].append(distanceFromCentroid)
                    else:
                        distancesDict[tp][videoFile].append(0)

                distancesDict[tp][videoFile] = distancesDict[tp][videoFile]/max(distancesDict[tp][videoFile])
                distancesDict[tp][videoFile].sort()

    with open(conf["dest_feature_file"], 'wb') as f:
        pickle.dump(distancesDict, f)
else: # Ha már van feature file akkor használjuk azt.
    with open(conf["source_feature_file"], 'rb') as f:
        distancesDict = pickle.load(f)

if conf["source_model_file"] == "none": # Ha még nincs feature file akkor csináljunk.

    pprint.pprint(distancesDict)

    # Get the length of any feature vector
    featureVectLength = len(distancesDict["real"][random.choice(list(distancesDict["real"]))])

    data = np.empty((1,featureVectLength + 1)) # This "empty" row will have to be deleted (add the +1 for Target column)

    for tp in distancesDict.keys():
        for videoFile,  features in distancesDict[tp].items():
            if tp == "real":
                features = np.append(features, 0)
            elif tp == "fake":
                features = np.append(features, 1)

            features = features[np.newaxis, :]
            data = np.append(data, features, axis=0)

    data = np.delete(data, (0), axis=0) # Delete first "empty" row

    featureVectColumns = []
    for i in range(featureVectLength):
        featureVectColumns.append("f" + str(i+1))

    featureVectColumns.append("target")

    df = pd.DataFrame(data, columns=featureVectColumns) #creates a new dataframe that's empty

    print(df)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(["target"], axis="columns"), df["target"], test_size=0.2)

    print("X_train length ", len(X_train))
    print("X_test length ", len(X_test))
    print("y_train length ", len(y_train))
    print("y_test length ", len(y_test))

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pickle.dump(model, open("RFmodel.pickle", 'wb'))

    score = model.score(X_test, y_test)
    print("test score = ", score)

    y_predicted = model.predict(X_test)

    cm = confusion_matrix(y_test, y_predicted)

    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot = True)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.savefig("confusionmatrix_score_" + str(score) + "_" + str(time.time()).split(".")[0] + ".png")