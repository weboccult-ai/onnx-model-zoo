import onnx
import torch
import onnxruntime as ort
import cv2
import numpy as np

class Detection:
    def __init__(self,conf_thresh=0.25,iou_thresh=0.2):
        """ This Class is used to perform inference on YOLOV8X model for face and person detection.

        Args:
            conf_thresh (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thresh (float, optional): NMS IOU threshold. Defaults to 0.5.
        """        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        self.session = ort.InferenceSession('yolov8x_person_face.onnx',providers=
                                ['CUDAExecutionProvider','CPUExecutionProvider','TensorrtExecutionProvider'],
                                verbose=True)
        
        self.input_names = [input.name for input in  self.session.get_inputs()]
        self.output_names = [output.name for output in  self.session.get_outputs()]
        
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
    def preprocess(self,img_path):
        """ This function is used to preprocess the image for inference.

        Args:
            img_path (str): Path to the image.
        
        Returns:
            original_image (np.array): Original image.
            processed_image (np.array): Processed image.
        """
        
        original_image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        
        input_image= original_image.copy()
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        
        input_image = cv2.resize(input_image,(self.input_shape[3],self.input_shape[2]))
        input_image = input_image / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis,:,:,:].astype(np.float32)
        
        return original_image,input_tensor
        
    def inference(self,img_path):
        """ This function is used to perform inference on the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            boxes (np.array): Bounding boxes.
            scores (np.array): Confidence scores.
            class_ids (np.array): Class ids.
            original_image (np.array): Original image.
        """        
        original_image,input_tensor = self.preprocess(img_path)
        self.img_height,self.img_width,_ = original_image.shape
        print(input_tensor.shape)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        boxes, scores, class_ids = self.process_output(outputs)
        boxes = boxes.astype(np.int32)
        return boxes, scores, class_ids, original_image
        
    def process_output(self,outputs):
        """ This function is used to process the output of YOLOV8X model for face and person detection.

        Args:
            outputs (list): List of outputs from the model.

        Returns:
            boxes (np.array): Bounding boxes.
            scores (np.array): Confidence scores.
            class_ids (np.array): Class ids.
        """        
        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:,4:],axis=1)
        predictions = predictions[scores > self.conf_thresh]
        scores = scores[scores > self.conf_thresh]
        
        if len(scores) == 0:
            return [], [], []
        
        class_ids = np.argmax(predictions[:,4:],axis=1)
        
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_thresh)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """ This function is used to extract bounding boxes from predictions.

        Args:
            predictions (np.array): Predictions from the model.

        Returns:
            boxes (np.array): Bounding boxes.
        """        
        
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)

        return boxes
    
    def rescale_boxes(self, boxes):
        """ This function is used to rescale the bounding boxes.

        Args:
            boxes (np.array): Bounding boxes.

        Returns:
            boxes (np.array): Rescaled bounding boxes.
        """
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_shape[2], self.input_shape[3], self.input_shape[2], self.input_shape[3]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def xywh2xyxy(self, x):
        """ This function is used to convert bounding boxes from xywh to xyxy format.

        Args:
            x (np.array): Bounding boxes in xywh format.

        Returns:
            y (np.array): Bounding boxes in xyxy format.
        """        
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    
    def nms(self,boxes, scores, iou_threshold):
        """ This function is used to perform non-maxima suppression.

        Args:
            boxes (np.array): Bounding boxes.
            scores (np.array): Confidence scores.
            iou_threshold (float): IOU threshold.

        Returns:
            keep_boxes (list): List of indices of boxes to keep.
        """        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes
    
    def compute_iou(self, box, boxes):
        """ This function is used to compute IOU.

        Args:
            box (list): Bounding box.
            boxes (list): List of bounding boxes.

        Returns:
            _type_: _description_
        """        
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou
    
if __name__ == '__main__':
    detection = Detection()
    detection.inference('weboccult_people.jpg')