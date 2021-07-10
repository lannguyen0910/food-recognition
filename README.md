# Online Food Detection using YOLOv5

![alt text](./demo/pipeline.png)

## 🌳 Folder Structure
```
YOLOv5-Online-Food-Detection
|
│   app.py                    # Flask server
|   modules.py                # inference stage, export result files, csv,...
|
└───api      
│   └─── ...
│   └─── api.py               # make request, update db
│   └─── secret.py            # get reponse 
|
└───model                     
│   └─── ...
│   └─── detect.py            # image detection
│   └─── video_detect.py      # video detection
|
└───static
│   └─── ...
│   └─── assets               # contain upload files, detection files
│   └─── css                  # custom css files, bootstrap
│   └─── js
|       └─── ...
│       └─── client.js        # custom js for templates
│       └─── chart.js         # nutrients analysys with charts
|
└───templates
│   └─── ...  
│   └─── index.html           # homepage, detect upload files
│   └─── url.html             # detect input URLs      
```

## 🌟 How to run locally (require GPU)
- Clone the repo
```
git clone https://github.com/lannguyen0910/YOLOv5-Online-Food-Detection
cd YOLOv5-Online-Food-Detection/
```
- Install dependencies
```
pip install -r requirements.txt
```
- Start the app normally 
```
python app.py --host=localhost:8000
```

## 🌟 Run using Google Colab with Ngrok
- Open notebook and follow the instructions [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JMH9vwvxmWy72yXxV2-niRUTW_J3PlQM?usp=sharing)
<!-- - (https://colab.research.google.com/drive/1SFDqNEQA9hrVA6zFn7wb0il-wV2Unou8?usp=sharing)-->

## 🌟 Train YOLOv5 
- Open notebook and follow the instructions [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PYMr192Y7Rc6SFLhq9ZVPQ64-9YM2fiF?usp=sharing)


## References
- YOLOv5 official repo: https://github.com/ultralytics/yolov5
- Inspiration from: https://ultralytics.com/yolov5
- Awesome object detection's custom template: https://github.com/kaylode/custom-template/tree/detection
