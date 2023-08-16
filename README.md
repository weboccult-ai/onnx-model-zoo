# model-zoo
Welcome to the onnx-model-zoo repository! This repo hosts a collection of machine learning models converted into ONNX, TensorRT and PyTorch formats, along with ready-to-use inference scripts and comprehensive demonstration code.

🔥 Key Features:
- Explore a variety of popular models in multiple formats for efficient inference.
- Access sample inference scripts for each model to jumpstart your projects.
- Dive into complete end-to-end demos showcasing model integration in real-world scenarios.

💡 Why Use This Repository?
Whether you're optimizing model deployment, benchmarking performance, or exploring inference strategies, this repository has you covered. Save time with pre-converted models and gain insights from ready-to-run demos.

### 🧠 Models
|Index|Model Name|Original URL|ONNX|FP32|FP16|INT8|TF-TRT|LICENCE|
|:-|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|[01](https://github.com/weboccult-ai/onnx-model-zoo/blob/main/MiVOLO_age_and_gender)|[MiVOLO](https://github.com/weboccult-ai/onnx-model-zoo/blob/main/MiVOLO_age_and_gender)|[🌐🌐🌐🌐🌐](https://github.com/WildChlamydia/MiVOLO/tree/main)|❌ |❌ |❌ |❌ |❌ |[Other](https://github.com/WildChlamydia/MiVOLO/blob/main/license/en_us.pdf)|

##### [01 MiVOLO Age and Gender Classification](https://github.com/weboccult-ai/onnx-model-zoo/blob/main/MiVOLO_age_and_gender)

```
python inference.py
```

![<b>Output</b>](https://github.com/weboccult-ai/onnx-model-zoo/blob/main/MiVOLO_age_and_gender/output.jpg)



#### 🤝 Contributing:
We welcome contributions! Help us expand the repository by adding new models, enhancing scripts, and sharing your use cases.

#### 📝 License:
This repository is licensed under the MIT License and for Model which you use please check the corresponding Licence.

#### 📬 Contact:
Have questions or suggestions? Reach out to us at [hiren@weboccult.com](mailto:hiren@weboccult.com) or connect on social media.

Start accelerating your model inference today with the onnx-model-zoo Repository!


#### Citations
```
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}
```