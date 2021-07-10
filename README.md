# **Online Food Detection using YOLOv5**

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
<summary>To increase the variety of dishes, we apply a classification model.</summary>
<br>
After testing and observing, we use a simple and effective model: EfficientNet. EfficientNet is proposed by Google and is one of the state-of-the-art models in this classification problem, and efficiency is also guaranteed. We apply the EfficientNet model source code from rwightman, we select the <b>EfficientNet-B4</b> version for retraining on the aggregated dataset. This model is used as an additional improvement to the YOLOv5 model in case the model detects a dish labeled as “Other Food”, only then EfficientNet is applied to predict the label again for this dish.
  
</details>
<details>
<summary>To increase the accuracy of the algorithm, we use the ensemble models technique.</summary>
For each image, models with different versions are used to predict, the results are then aggregated using the "weighted box fusion" method to give the final result.
<br>
  
</details>
<details>
<summary>To increase users' interactivity with the application.</summary>
When a dish is predicted, we provide more information about the nutritional level of that dish to the user. This information is queried from the application's database, which will be periodically updated from the Edamam API - an API that allows querying the nutrition of a dish by dish name. When doing prediction, the nutrition information will be saved along with the dish name under <b>CSV</b> format. We then fetch the CSV file on the client site to proceed drawing nutritrion statistics chart using <a href="https://github.com/chartjs/Chart.js">Chart.js</a> library. There are a total of 2 chart types, which appear when the user clicks on that chart type.
</details>
<br>

## 📙 **References**
- YOLOv5 official repo: https://github.com/ultralytics/yolov5
- Inspiration from: https://ultralytics.com/yolov5
- Awesome object detection's custom template: https://github.com/kaylode/custom-template/tree/detection
- Edamam API: https://developer.edamam.com/food-database-api-docs
- Chart.js: https://github.com/chartjs/Chart.js

