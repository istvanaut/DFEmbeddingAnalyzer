testDict = {"real" : {"0_video1": {}, "1_video2": {}}, "fake": {"video500_0":{}, "video501_1":{}}}

for fakeVideo in testDict["fake"].keys():
    print(fakeVideo, fakeVideo.split("_")[-1] + "_")
    # realVideo = testDict["real"][fakeVideo.split("_")[-1]]

    result = [i for i in testDict["real"].keys() if i.startswith(fakeVideo.split("_")[-1] + "_")]

    print(result)