# **Food detection on android with YOLOv5 and NCNN**

Ncnn deployment on android, support YOLOv5s for food detection

# **App usage**
- After running the app, there's a ```YoloV5s``` button. Click the button to start.
- By default we'll "livestream" based on our back camera. The model'll detect food objects and write bboxes + labels (Accuracy is still low) on our view display in the middle of the screen.
- Click ```Photo``` button to choose the photo we want to detect. The result will display in our image view. Click that view to switch to default.
- Click ```Video``` button to choose the video we want to detect. We can speed up the video result with the speed bar at the bottom. **Recommend dragging it to the maximum for easy monitoring**.

# **Notes**
- I remove many models in original repo and remain only YOLOv5 model.
- Modified some code and add more assets to make it look like a **food detection app**.
- If you use my pytorch weights from [here](https://drive.google.com/drive/folders/1gL16SVnLeI7cUnBMeK54JwKKOWiOybrc?usp=sharing) and export it using my [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nf0lLo6e2nMAt_AtDNoHmeXzdAB9kxsj?usp=sharing), then you can run the app on Android Studio.
- Can only use YOLOv5s.
- First [apk](https://github.com/lannguyen0910/food-detection-yolov5/releases/tag/1.0) release.
- The accuracy is still low, we'll update gradually.
- The app is for learning purpose only!

# **Credits** 
* [Original implementation](https://github.com/cmdbug/YOLOv5_NCNN)
* [Reference article](https://zhuanlan.zhihu.com/p/279288150)
* [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5) 
* [NCNN by Tencent](https://github.com/tencent/ncnn)
