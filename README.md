# 🍔🍟🍗 **Online Food Detection using YOLOv5** 🍞🍖🍕

![alt text](./demo/pipeline.png)

## 🌳 **Folder Structure**
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
│   └─── index.html           # upload files' page
│   └─── url.html             # input URLs' page    
```
<br>

## 🌟 **How to run locally (require GPU)**
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

## 🌟 **Run using Google Colab with Ngrok**
- Open notebook and follow the instructions [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JMH9vwvxmWy72yXxV2-niRUTW_J3PlQM?usp=sharing)
<!-- - (https://colab.research.google.com/drive/1SFDqNEQA9hrVA6zFn7wb0il-wV2Unou8?usp=sharing)-->

## 🌟 **Train YOLOv5** 
- Open notebook and follow the instructions [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PYMr192Y7Rc6SFLhq9ZVPQ64-9YM2fiF?usp=sharing)
<br>

## 🌟 **Datasets**
<details>
<summary>To train the food detection model, we survey the following datasets:</summary>
  
  - <a href="https://storage.googleapis.com/openimages/web/index.html">Open Images V6-Food</a>: Open Images V6 is a huge dataset from Google for Computer Vision tasks. To solve our problem, we extracted from a large dataset on food related labels. The extracted set includes 18 labels with more than 20,000 images.
  - <a href="http://foodcam.mobi/dataset.html">School Lunch Dataset</a>: includes 3940 photos of a lunch of Japanese high school students, taken at the same frontal angle with the goal of assessing student nutrition. Labels consist of coordinates and types of dishes are attached and divided into 21 different dishes, in the dataset there is also a label "Other Food" if the dishes do not belong to the remaining 20 dishes.
  - <a href="https://drive.google.com/drive/folders/1PLrsJBS1EtJLY6RpEriASTfQ5WRlPQAt?usp=sharing">Vietnamese Food</a>: a self-collected dataset on Vietnamese dishes, including 10 simple dishes of our country such as: Pho, Com Tam, Hu Tieu, Banh Mi,... Each category has about 20-30 images, divided 80-20 for training and evaluation.
  
We aggregate all the above datasets to proceed training. Dishes that appear in different sets will be grouped into one to avoid duplication. After aggregating, a large data set of 60,305 images with 44 different foods from all regions of the world.
<br>
  
In addition, we find that if we expand the problem to include classification, the dataset will increase significantly. Therefore, to further enhance the diversity of dishes, we collect additional datasets to additionally train a classification model:
  - <a href="http://www.ub.edu/cvub/mafood121/">MAFood-121</a>: consisting of 21,175 training image samples. The dishes are selected from the top 11 most popular cuisines in the world according to Google Trends statistics, these cuisines come from many countries around the world, especially Vietnam. For each type of cuisine, 11 typical traditional dishes are selected. The dataset has a total of 121 different types of dishes, each belonging to at least 1 of 10 food categories: Bread, Eggs, Fried, Meat, Noodles, Rice, Seafood, Soup, Dumplings, and Vegetables . 85% of the images are used for training and the remaining 15% for evaluation.
  - <a href="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/">Food-101</a>: includes 101 different types of dishes, with 101,000 sets of photos. For each dish, 250 images were used as test images and the remaining 750 images were used for training. The training images in this set still have a lot of noise, sometimes the colors are too sharp or some of the data samples are mislabeled, these noises are intentional by the author (mentioned in the study).
  
We also perform the aggregation of the two data sets above into one. The new set includes <b>93,748 training images</b> and <b>26,825 evaluation images</b> with a total of <b>180 different dishes</b>. It can be seen that the number of dishes has increased significantly, if the model detects a dish labeled "Other Food", the classification model will be applied to this dish and classified again.
</details>

## 🌟 **YOLOv5 models**
We use <b>4 YOLOv5 versions (s, m, l and x)</b> and their pre-trained weights to train on our problem. The model's source code is inherited from the <a href="https://github.com/ultralytics/yolov5">Ultralytics</a> source code repo, and the training and data processing steps are reinstalled by us using Pytorch. When training, data augmentation methods are applied to minimize the model's easy overfit.
<br>

## 🌟 **Additional Methods**
<details>
<summary>To increase the variety of dishes, we apply a classification model:</summary>
<br>
After testing and observing, we use a simple and effective model: EfficientNet. EfficientNet is proposed by Google and is one of the state-of-the-art models in this classification problem, and efficiency is also guaranteed. We apply the EfficientNet model source code from rwightman, we select the <b>EfficientNet-B4</b> version for retraining on the aggregated dataset. This model is used as an additional improvement to the YOLOv5 model in case the model detects a dish labeled as “Other Food”, only then EfficientNet is applied to predict the label again for this dish.
  
</details>
<details>
<summary>To increase the accuracy of the algorithm, we use the ensemble models technique:</summary>
<br>
  
For each image, models with different versions are used to predict, the results are then aggregated using the "weighted box fusion" method to give the final result.
  
</details>
<details>
<summary>To increase users' interactivity with the application:</summary>
<br>
  
When a dish is predicted, we provide more information about the nutritional level of that dish to the user. This information is queried from the application's database, which will be periodically updated from the Edamam API - an API that allows querying the nutrition of a dish by dish name. When doing prediction, the nutrition information will be saved along with the dish name under <b>CSV</b> format. We then fetch the CSV file on the client site to proceed drawing nutritrion statistics chart using <a href="https://github.com/chartjs/Chart.js">Chart.js</a> library. There are a total of 2 chart types, which appear when the user clicks on that chart type.
</details>

## 🍱 **Sample Results**
<p float="left">
  <img src="demo/2.jpg" width="100%" /> 
  <img src="demo/1.jpg" width="100%" />
  <img src="demo/3.jpg" width="100%" />
  <img src="demo/4.jpg" width="100%" />
  <img src="demo/5.jpg" width="100%" /> 
  <img src="demo/6.jpg" width="100%" />
</p>

## 📝 **Appendix**
<details open>
<summary>mAP of each model:</summary>
  <br>
  
Due to hardware limitations, only two models YOLOv5s and YOLOv5m were trained on the entire dataset, the other two models, YOLOv5l and YOLOv5x, were trained on only a small set of data.
  <br>
  
- Valset:

| Models  | Image Size | mAPsmall | mAPmedium | mAPlarge | mAP@0.5:0.95 |
| ------- | :--------: | :------: | :-------: | :------: | :----------: |
| YOLOv5s |  640x640   |    5     |   29.7    |   31.1   |     31.5     |
| YOLOv5m |  640x640   |   8.6    |   30.1    |   30.9   |     32.3     |
| YOLOv5l |  640x640   |   16.7   |   51.9    |   52.0   |     56.0     |
| YOLOv5x |  640x640   |   18.4   |    50     |   50.9   |     55.1     |

---

- Testset:

| Models  | Image Size | mAPsmall | mAPmedium | mAPlarge | mAP@0.5:0.95 |
| ------- | :--------: | :------: | :-------: | :------: | :----------: |
| YOLOv5s |  640x640   |   6.3    |   28.4    |   30.3   |     30.5     |
| YOLOv5m |  640x640   |   7.5    |   28.7    |   30.6   |     30.9     |
| YOLOv5l |  640x640   |   38.4   |   46.8    |   57.9   |     57.5     |
| YOLOv5x |  640x640   |   55.0   |   48.9    |   59.8   |     55.1     |
</details>
<br>

<details>
  <summary>The loss and the learning rate through the loops of the 2 models YOLOv5l and YOLOv5x , have different values due to different image sizes:</summary>
  <br>
  
  <img src="demo/yolov5l,x.PNG" width="100%" /> 
</details>

<details>
  <summary>Losses and mAP of 2 models YOLOv5l and YOLOv5x:</summary>
  <br>
  
  <img src="demo/yolov5l,x_val.PNG" width="100%" /> 
  <br>
  
The results on valset are quite similar between the two models, the highest mAPsmall value belongs to YOLOv5x, because it was trained on a larger image. However, due to the limitation of Google Colab, and the number of parameters of these two models is quite high, the training has not been completed, the model has not really converged. We found that if the training was longer, there would be better results.
  
</details>

<details>
  <summary>The YOLOv5s and YOLOv5m models were trained on the entire large dataset and converged after 100 epochs, both trained with the same configuration. The result of the training process of YOLOv5s and YOLOv5m models are as follows:</summary>
  <br>
  

  <img float="left" src="demo/yolov5s,m.png" width="50%"/>
  <img float="right" src="demo/yolov5s,m_val.png" width="50%"/>
  <br>
  
It was observed that the YOLOv5m model converged faster than the YOLOv5s, and also overfitted the data more quickly. The highest mAPmedium and mAPlarge values of the two models are also quite close, however YOLOv5m outperforms the other model in detecting small objects.

</details>
<br>

We conclude that the learned models are quite good compared to such huge data with such a variety of dishes, and their complexity is also guaranteed to be suitable for practical applications. We show that the system is scalable and has high practical significance for the people of Vietnam in particular and the world in general.
<br>

## 💡 **Further Improvements**
In the process of implementing the project, our team encountered many difficulties, thanks to the research and study of many sources, we can solve important issues. However, still have to accept shortcomings in terms of hardware as well as resources. In addition, the team still see the possibility of improving the system if the training dataset is processed more carefully, such as:
- We have yet to solve the problem of messy data labels (with one sample labeling the dish, another labeling the ingredients of the dish) causing the score to be not really accurate.
- In addition, <b>when using Ngrok on Google Colab</b>, the team encountered an error that FFMPEG on OpenCV does not support the <b>.mp4</b> format, so the video will not be displayed on the website. To fix this we had to recompile OpenCV, and because that took a long time, the team still couldn't finish for now. However, this error was fixed when running at localhost.
- Finally, due to the limited knowledge of web development, the team still encountered some difficulties. For example, in dealing post requests using ajax, fetch, ... from client to server, which leads to certain limitations.

In the future, we hope to solve all of the above problems and continue to develop the application on mobile device like Android.
<br>

## 📙 **References**
- YOLOv5 official repo: https://github.com/ultralytics/yolov5
- Inspiration from: https://ultralytics.com/yolov5
- Awesome object detection's custom template: https://github.com/kaylode/custom-template/tree/detection
- Edamam API: https://developer.edamam.com/food-database-api-docs
- Chart.js: https://github.com/chartjs/Chart.js
