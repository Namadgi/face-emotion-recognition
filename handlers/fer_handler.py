import io
import os
import cv2
import json
import base64
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

import torch
from ts.torch_handler.base_handler import BaseHandler

class FERHandler(BaseHandler):
    onnx_model = None
    INPUT_SIZE = (224, 224)

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        if data is None:
            return data

        for row in data:
            data = row.get('data') or row.get('body')
        
        if isinstance(data, dict):
            data: dict = data['instances'][0]
            # Download file
            if 'b64' not in data.keys():
                token = data['token']
                bucket_name = data['bucket_name']
                object_name = data['object_name']
                os.system(
                    f'curl -X GET ' +
                    f'-H "Authorization: Bearer {token}" -o {object_name} '
                    f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"'
                )

                cap = cv2.VideoCapture(object_name)
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                t_frame = n_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame)
                res = False
                while not res:
                    res, frame = cap.read()
                cap.release()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
        
            b64_str = data['b64']
            data = base64.b64decode(b64_str)
            
        return ImageOps.exif_transpose(Image.open(
            io.BytesIO(data)
        ))

    def inference(self, full_image):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        with torch.no_grad():
            y = self.model(full_image, self.device)
        return y

    def postprocess(self, face = None, code = 0):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        
        if code != 0: 
            err_descrs = [
                'No face, too small or bad quality',
                'Found more than ONE face',
                'Face is not centered',
            ]   
            return [{
                'code': code,
                'description': err_descrs[code - 2],
                'result': 'ERROR',
            }]
        
        img = np.array(face)
        img = cv2.resize(img, self.INPUT_SIZE)
        img = img.astype(np.float32)
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
        img = np.expand_dims(img, axis=0)
    
        results_ort = self.onnx_model.run(["emotion_preds"], {"x": img})[0]
        label = int(np.argmax(results_ort[0]))

        classes = [
            'Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]

        return [{
            'code': code,
            'description': 'Successful check',
            'result': classes[label],
        }]
        

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if self.onnx_model is None:
            self.load_model()
        
        full_image = self.preprocess(data)
        res = self.inference(full_image)
        if isinstance(res, int):
            return self.postprocess(code=res)
        return self.postprocess(face=res)
    
    def load_model(self):
        self.onnx_model = ort.InferenceSession("mobnet.onnx", providers=["CPUExecutionProvider"])

        