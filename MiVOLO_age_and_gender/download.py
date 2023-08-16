import requests

# Download the file from `https://weboccult-models.s3.us-west-2.amazonaws.com/onnx-model-zoo/MiVOLO/yolov8x_person_face.onnx' 
# under the name `yolov8x_person_face.onnx'

model = 'yolov8x_person_face.onnx'
url = 'https://weboccult-models.s3.us-west-2.amazonaws.com/onnx-model-zoo/MiVOLO/yolov8x_person_face.onnx'
r = requests.get(url, allow_redirects=True)
open(model, 'wb').write(r.content)

url = "https://weboccult-models.s3.us-west-2.amazonaws.com/onnx-model-zoo/MiVOLO/modified_mivolo_age_gender.onnx"
model = "modified_mivolo_age_gender.onnx"
r = requests.get(url, allow_redirects=True)
open(model, 'wb').write(r.content)
