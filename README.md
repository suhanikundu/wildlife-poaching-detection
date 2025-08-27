# 🦌 Wildlife Poacher Detection and Alerting system in Real-time

This repository contains code for training and detecting **poachers in the wild** using a custom **YOLOv3** model through webcam or external device.  


I have made the `poacher-Implementation.ipynb` file private to avoid misuse. Contact me at 

📩 **suhani.kundu2406@.com** for complete access.  

---

## 📦 Resources  
| 📒 Colab Notebook | 🧠 Complete Folder |  
|-------------------|---------------------|  
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XL2NzO_WaWeyQVlVhmnqNG82vjEz008Y?usp=sharing) | [Project Folder](https://drive.google.com/drive/folders/1vvhg2fayBCQngZidOL04CA05Z8gl83tB?usp=sharing) |  

---

## 💡 Sample Inputs  
| 1.jpg | 2.jpg | pic1.jpg | pic2.jpg |  
|-------|-------|----------|----------|  
| ![input1](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/1.jpg) | ![input2](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/2.jpg) | ![input3](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/pic1.jpg) | ![input4](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/pic2.jpg) |  

---

## 🧠 Sample Outputs  
| 1.jpg | 2.jpg | pic1.jpg | pic2.jpg |  
|-------|-------|----------|----------|  
| ![output1](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/Screenshot%202025-08-27%20205859.png) | ![output2](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/Screenshot%202025-08-27%20205958.png) | ![output3](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/Screenshot%202025-08-27%20210019.png) | ![output4](https://github.com/suhanikundu/wildlife-poaching-detection/blob/main/test_images/Screenshot%202025-08-27%20205724.png) |  

---

## 📂 Files Required
- 📌 Darknet repository  
- 📌 Labeled custom dataset  
- 📌 Custom `.cfg` file  
- 📌 `obj.data` and `obj.names` files  
- 📌 `train.txt` file (optional: `test.txt`)  

🎥 Tutorial reference: [YouTube Video by The AI Guy](https://www.youtube.com/watch?v=10joRJt39Ns)  

🔗 Download YOLO weights:  
- [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)  
- [YOLOv3-Tiny Weights](https://pjreddie.com/media/files/yolov3-tiny.weights)  

---

## ⚡ Colab Hack ⭐  
If you’re using free **Google Colab** and face disconnect issues, try this trick:  

👉 Step 1: In Colab, press **CTRL + SHIFT + I** (Inspect element)  
👉 Step 2: Go to **Console tab** and paste the following code:  

```js
function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60000)

```
## Contributors

<a href="https://github.com/suhanikundu/wildlife-poaching-detection/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=suhanikundu/wildlife-poaching-detection" />
</a>
