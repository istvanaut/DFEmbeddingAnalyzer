import numpy as np
import cv2
import os
from json_minify import json_minify
import json
import face_recognition
import pickle
import pprint
import matplotlib.pyplot as plt
import insightface

conf = json.loads(json_minify(open("conf.json").read()))
enc = None
agesDict = {"real": {}, "fake": {}}
genderDict = {"real": {}, "fake": {}}

USE_INSIGHTFACE = bool(conf["USE_INSIGHTFACE"])
FRAME_LIMIT = 240 # Break loop once processed this frame from the video
VIDEO_LIMIT = None # Only process this many videos from a folder. (enter None for ALL videos)

if USE_INSIGHTFACE:
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id = -1, nms=0.4)

if conf["source_enc_file"] == "none": # Ha még nincs enc file akkor csináljunk.
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
            agesDict[tp][videoFile] = []
            genderDict[tp][videoFile] = []

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
                            agesDict[tp][videoFile].append(faceEncoding[0].age)
                            genderDict[tp][videoFile].append(faceEncoding[0].gender)
                        else:
                            enc[tp][videoFile].append(None)
                            agesDict[tp][videoFile].append(-3)
                            genderDict[tp][videoFile].append(None)
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
                centroid = sumOfEncodings / numberOfEncodings
            except:
                print(f"\nERROR! Coudn't compute centroid for {videoFile} (division by zero)")
                print(f"List of encodings: {listOfEncodings}\n")

            enc[tp][videoFile].append({videoFile + "_centroid": centroid})

    with open(conf["dest_enc_file"], 'wb') as f:
        pickle.dump(enc, f)
else: # Ha már van enc file akkor használjuk azt.
    with open(conf["source_enc_file"], 'rb') as f:
        enc = pickle.load(f)
    countOfRealPlusFakeVideos = len(enc["real"].keys()) + len(enc["fake"].keys())

# Calculate distances from centroid

print("\nStarting to calculate distances from centroid...\n")

distancesDict = {"real": {}, "fake": {}}
videoCount = 0
for tp in enc.keys():
    for videoFile, encodingsAndCentroid in enc[tp].items():
        videoCount += 1
        print(f"Calculating distance from centroid for video {videoCount} out of {countOfRealPlusFakeVideos}")
        distancesDict[tp][videoFile] = []
        centroid = encodingsAndCentroid[-1][videoFile + "_centroid"]
        for encoding in encodingsAndCentroid[:-1]:
            if encoding is not None:
                distanceFromCentroid = np.linalg.norm(centroid-encoding)
                distancesDict[tp][videoFile].append(distanceFromCentroid)
            else:
                distancesDict[tp][videoFile].append(-0.3)

print("\nStarting plotting DISTANCES real-fake pairs...\n")

videoCount = 0
for fakeVideo in distancesDict["fake"].keys():

    realVideo = [i for i in distancesDict["real"].keys() if i.startswith(fakeVideo.split(".")[0].split("_")[-1] + "_")][0]

    fakeVideoDistances = distancesDict["fake"][fakeVideo]
    realVideoDistances = distancesDict["real"][realVideo]

    videoCount += 1

    print(f"Plotting videos {realVideo} and {fakeVideo}")
    try:
        plt.plot(fakeVideoDistances, marker=".", linestyle='None', color='red')
        plt.plot(realVideoDistances, marker="*", linestyle='None', color='green')
        ymax = max([max(fakeVideoDistances)*1.2, max(realVideoDistances)*1.2])
        plt.ylim(-0.5, ymax)
        plt.savefig(fakeVideo.split(".")[0] + "_" + realVideo.split(".")[0] + '_DISTANCE_pair.png')
        plt.clf()
    except:
        print(f"\nWARNING! Coudn't compute DISTANCE ymax for {fakeVideo, ' and ' , realVideo} (zero elements)\n")

print("\nStarting plotting AGES for real-fake pairs...\n")

videoCount = 0
for fakeVideo in agesDict["fake"].keys():

    realVideo = [i for i in agesDict["real"].keys() if i.startswith(fakeVideo.split(".")[0].split("_")[-1] + "_")][0]

    fakeVideoAges = agesDict["fake"][fakeVideo]
    realVideoAges = agesDict["real"][realVideo]

    videoCount += 1

    print(f"Plotting videos {realVideo} and {fakeVideo}")
    try:
        plt.plot(fakeVideoAges, marker=".", linestyle='None', color='red')
        plt.plot(realVideoAges, marker="*", linestyle='None', color='green')
        ymax = max([max(fakeVideoAges)*1.2, max(realVideoAges)*1.2])
        plt.ylim(-0.5, ymax)
        plt.savefig(fakeVideo.split(".")[0] + "_" + realVideo.split(".")[0] + '_AGES_pair.png')
        plt.clf()
    except:
        print(f"\nWARNING! Couldn't compute AGE ymax for {fakeVideo, ' and ' , realVideo} (zero elements)\n")


print("\nComparing GENDERS for real-fake pairs...\n")

for fakeVideo in genderDict["fake"].keys():

    realVideo = [i for i in genderDict["real"].keys() if i.startswith(fakeVideo.split(".")[0].split("_")[-1] + "_")][0]

    fakeVideoGenders = genderDict["fake"][fakeVideo]
    realVideoGenders = genderDict["real"][realVideo]

    totalFrames = len(fakeVideoGenders)

    genderDifference = 0
    for i, fakeCurrentGender in enumerate(fakeVideoGenders):
        realCurrentGender = realVideoGenders[i]
        if  (fakeCurrentGender != realCurrentGender) and (fakeCurrentGender is not None) and (realCurrentGender is not None):
           genderDifference += 1

    print(f"Gender difference between {realVideo} and {fakeVideo} is {genderDifference} / {totalFrames}")

    if genderDifference/totalFrames > 0.3:
        print("Fake: " + str(fakeVideoGenders))
        print("Real:" + str(realVideoGenders))




# print("\nStarting plotting single plots...\n")
#
# videoCount = 0
# for tp in distancesDict.keys():
#     for videoFile, distances in distancesDict[tp].items():
#         videoCount += 1
#         print(f"Plotting video {videoCount} out of {countOfRealPlusFakeVideos}")
#         try:
#             plt.plot(distances, marker=".", linestyle='None')
#             ymax = max(distances)*1.2
#             plt.ylim(-0.5, ymax)
#             plt.savefig(videoFile.split(".")[0] + '.png')
#             plt.clf()
#         except:
#             print(f"\nWARNING! Coudn't compute ymax for {videoFile} (zero element)\n")



# print("\nPlotting all plots on the same figure...\n")
# ideoCount = 0
# for tp in distancesDict.keys():
#     for videoFile, distances in distancesDict[tp].items():
#         videoCount += 1
#         print(f"Plotting video {videoCount} out of {countOfRealPlusFakeVideos}")
#         try:
#             if tp == "real":
#                 plt.plot(distances, marker="*", linestyle='None', color='green')
#             else:
#                 plt.plot(distances, marker=".", linestyle='None', color='red')
#         except:
#             print(f"\nWARNING! Coudn't compute DISTANCE ymax for {videoFile} (zero element)\n")
#
# plt.ylim(ymin = -0.5)
# plt.savefig('all.png')
# plt.clf()
