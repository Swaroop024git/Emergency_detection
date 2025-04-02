# 🚨 Emergency Detection using Emotions 🚀  

🔗 **GitHub Repository:** [Emergency Detection](https://github.com/Swaroop024git/Emergency_detection.git)  

Welcome to **Emergency Detection**, a cutting-edge **ROS2-powered AI system** that detects emergency situations by analyzing emotions in real time! 🎭🆘  

This system intelligently combines multiple detected emotions to determine if a passenger is in **distress** and requires immediate attention. Whether it's **fear, sadness, or unusual emotional patterns**, this detector ensures that help arrives on time! 🚑  

---

## 🌟 Features  

✅ **Real-Time Emotion-Based Emergency Detection** 🚦  
✅ **Deep Learning with CNN** 🧠  
✅ **ROS2 Integrated** for seamless robotic applications 🤖  
✅ **Live Visualization using OpenCV** 🎨  
✅ **Health Monitoring & Alert System** 🚨  

---

## 🛠️ Setup & Installation  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/Swaroop024git/Emergency_detection.git
cd Emergency_detection
```

### **2️⃣ Install Dependencies**  
Ensure you have the required dependencies installed:  
```sh
pip install torch torchvision numpy opencv-python pillow torchinfo
sudo apt install ros-foxy-cv-bridge  # (or your ROS2 version)
```

### **3️⃣ Model File**  
Download the pre-trained model and place it in:  
```
interior_monitoring/src/interior_monitoring/data/emotion_cnn.pth
```

### **4️⃣ Run the Emergency Detector Node**  
```sh
ros2 run emergency_detector emergency_detector
```

---

## 🎯 How It Works  

1️⃣ Captures **real-time images** from the camera 📷  
2️⃣ Analyzes emotions using a **deep learning model** 🧠  
3️⃣ Monitors **patterns of fear, sadness, and distress** 😨  
4️⃣ **Triggers an emergency alert** if a critical emotional combination is detected 🚨  
5️⃣ Publishes results via **ROS2 topics** and displays real-time status 🚦  

---

## 📡 ROS2 Topics  
| **Topic**            | **Type**                | **Description** |
|----------------------|------------------------|----------------|
| `/image_raw`        | `sensor_msgs/Image`    | Input image stream |
| `/emotion_status`   | `std_msgs/String`      | Detected emotion |
| `/passenger_health` | `std_msgs/Bool`        | Emergency alert 🚨 |
| `/emotion_probs`    | `std_msgs/Float32MultiArray` | Emotion probabilities |
| `/emotion_marker`   | `visualization_msgs/Marker` | RViz visualization |

---


## 📌 Roadmap  
- [ ] Improve **emergency classification accuracy** 🎯  
- [ ] Optimize **response time** for real-world applications ⏳  
- [ ] Integrate **automated alerts to emergency contacts** 📞  
- [ ] Deploy in **autonomous vehicles & smart environments** 🚗🏠  

---

## 🤝 Contributing  
Want to improve this project? Found a bug? PRs are welcome! 🎉  

📬 **Let's Connect:**  
💬 Open an Issue | 🌟 Star the Repo | 🚀 Contribute  

---

## ⚖️ License  
This project is **MIT Licensed**. Use it, modify it, and help make the world safer! 🌍🚀  

---

💡 *"AI isn't just about intelligence, it's about saving lives!"* ❤️🚀
