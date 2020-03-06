import cv2
import insightface
import face_recognition

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id = -1, nms=0.4)
img1 = cv2.imread("C:\\Users\\IstvanLaptop\\PycharmProjects\\DFEmbeddingAnalyzer\\akibmagvog_300.jpeg")
img2 = cv2.imread("C:\\Users\\IstvanLaptop\\PycharmProjects\\DFEmbeddingAnalyzer\\mvuidgordx_60.jpeg")

face1if = model.get(img1)

face1fr = face_recognition.face_encodings(img1)