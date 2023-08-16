import torch
import onnxruntime as ort
import numpy as np
import cv2
from easydict import EasyDict as edict

from detection import Detection

class AgeAndGender:
    
    def __init__(self) -> None:
        """ 
        Load the model and initialize the session.
        """        
        self.ort_sess = ort.InferenceSession('modified_mivolo_age_gender.onnx',providers=
                                ['CUDAExecutionProvider','CPUExecutionProvider','TensorrtExecutionProvider'],
                                verbose=True)

        self.output_nodes= [node.name for node in self.ort_sess.get_outputs()]
        
        meta = {'min_age': 1, 'max_age': 95, 'avg_age': 48.0, 'num_classes': 3, 'in_chans': 3, 'with_persons_model': False, 'disable_faces': False,
                'use_persons': False, 'only_age': False, 'num_classes_gender': 2, 'use_person_crops': False, 'use_face_crops': True}
        self.meta = edict(meta)
        
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.output_name = self.ort_sess.get_outputs()[0].name
        
        self.target_size = 224
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.crop_pct = 0.96

    def inference(self,bboxes,original_image,class_ids):
        """ This function is used to perform inference on the image.q

        Args:
            bboxes (np.array): Bounding boxes.
            original_image (np.array): Original image.
            class_ids (np.array): Class ids.

        Returns:
            persons(list): list of people containing {'age':age,'gender':gender,'bbox':bbox}
            original_image (np.array): Original image.
        """        
        persons = []
        for bbox,class_id in zip(bboxes,class_ids):
            if class_id != 1:
                continue
            face = self.prepare_face(bbox,original_image)
            result = self.ort_sess.run(None, {self.input_name: face})
            age,gender = self.post_process(result)
            persons.append({'age':age,'gender':gender,'bbox':bbox})
        return persons,original_image
        
    def prepare_face(self,bbox,original_image):
        """ This function is used to prepare the face for inference.

        Args:
            bbox (np.array): Bounding box.
            original_image (np.array): Original image.

        Returns:
            img (np.array): Processed image.
        """        
        img = original_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        
        img = self.class_letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - self.mean) / self.std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
        
    def class_letterbox(self, im, new_shape=(224, 224), color=(0, 0, 0), scaleup=True):
        """ This function is used to letterbox the image.

        Args:
            im (_type_): _description_.
            new_shape (tuple, optional): shape of the image. Defaults to (224, 224).
            color (tuple, optional): color of the image. Defaults to (0, 0, 0).
            scaleup (bool, optional): scale up the image. Defaults to True.

        Returns:
            im (np.array): Processed image.
        """        
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
            return im

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        # ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def post_process(self,output):
        """ This function is used to post process the output of the model.

        Args:
            output (np.array): Output of the model.

        Returns:
            age (int): Age of the person.
        """        
        
        gemm_result = output[0]
        add_result = output[1]

        reduce_max = np.max(add_result, axis=1, keepdims=False)
        mul = np.multiply(reduce_max, 0.5)
        output = np.add(mul, gemm_result)[0]
        
        age = output[2] * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
        age = int(age)
        gender = "M" if output[0] > output[1] else "F"
        
        return age,gender
    
if __name__ == '__main__':
    age_and_gender_model = AgeAndGender()
    detection_model = Detection()

    image_path = "input.jpg"
    boxes, scores, class_ids,original_image = detection_model.inference(image_path)

    persons,original_image = age_and_gender_model.inference(boxes,original_image,class_ids)
    for person in persons:
        if person['gender'] == 'F':
            color = (255,0,0)
        else:
            color = (0,0,255)
        original_image = cv2.rectangle(original_image,(person['bbox'][0],person['bbox'][1]),(person['bbox'][2],person['bbox'][3]),color,10)
        original_image = cv2.putText(original_image,str(person['age'])+"."+str(person['gender']),(person['bbox'][0],person['bbox'][1]),cv2.FONT_HERSHEY_COMPLEX,3,color,10)
    cv2.imwrite('output.jpg',original_image)