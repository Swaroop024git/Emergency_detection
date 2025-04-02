import rclpy
import os
import cv2
import torch
import numpy as np
from rclpy.node import Node
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Bool, Float32MultiArray
from visualization_msgs.msg import Marker
from PIL import Image as PILImage
from torchvision import transforms

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionDetector(Node):
    def __init__(self):
        super().__init__('emotion_detector')
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # Health monitoring parameters
        self.sick_threshold = 3
        self.emotion_buffer = []
        
        # ROS2 Setup
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.emotion_pub = self.create_publisher(String, '/emotion_status', 10)
        self.health_pub = self.create_publisher(Bool, '/passenger_health', 10)
        self.prob_pub = self.create_publisher(Float32MultiArray, '/emotion_probs', 10)
        self.marker_pub = self.create_publisher(Marker, '/emotion_marker', 10)
        self.bridge = CvBridge()
        
        # Visualization parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'neutral': (0, 255, 0),    # Green
            'happy': (255, 255, 0),    # Yellow
            'sad': (0, 0, 255),        # Red
            'fear': (255, 0, 0),       # Blue
            'default': (255, 255, 255) # White
        }
        
        # Model loading
        self.model = None
        self.transform = None
        self.load_model()

    def load_model(self):
        try:
            model_path = os.path.expanduser(
                '~/interior_monitoring/src/interior_monitoring/data/emotion_cnn.pth'
            )
            
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found: {model_path}")
                raise FileNotFoundError("Model file missing")

            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            self.model = EmotionCNN(num_classes=7)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval().to(self.device)
            self.get_logger().info("Model loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Model initialization failed")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            display_image = cv_image.copy()
            
            # Process image
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_image)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            # Get prediction
            _, predicted = torch.max(outputs, 1)
            emotion = self.class_idx_to_label(predicted.item())
            
            # Update displays
            self.update_health_status(emotion)
            self.visualize_openCV(display_image, emotion, probs)
            self.publish_marker(emotion)
            self.publish_probs(probs)
            
            # Publish results
            self.publish_results(emotion)

        except Exception as e:
            self.get_logger().error(f'Processing failed: {str(e)}')
            cv2.destroyAllWindows()

    def visualize_openCV(self, image, emotion, probs):
        # Draw emotion text
        color = self.colors.get(emotion, self.colors['default'])
        cv2.putText(image, f"Emotion: {emotion}", (10, 30), 
                   self.font, 0.7, color, 2, cv2.LINE_AA)
        
        # Draw probability bars
        bar_height = 20
        for i, prob in enumerate(probs):
            label = self.class_idx_to_label(i)
            bar_width = int(prob * 200)
            cv2.rectangle(image, (10, 40 + i*30), 
                         (10 + bar_width, 40 + i*30 + bar_height),
                         self.colors.get(label, (255, 255, 255)), -1)
            cv2.putText(image, f"{label}: {prob:.2f}", (20, 55 + i*30), 
                       self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Show health warning
        if self.is_sick():
            cv2.putText(image, "PASSENGER DISTRESS DETECTED!", (10, 250), 
                       self.font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Emotion Monitor", image)
        cv2.waitKey(1)

    def publish_marker(self, emotion):
        marker = Marker()
        marker.header.frame_id = "camera_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = emotion
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0 if emotion in ['sad', 'fear'] else 0.0
        marker.color.g = 1.0 if emotion == 'neutral' else 0.0
        marker.color.b = 1.0 if emotion == 'happy' else 0.0
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.5
        self.marker_pub.publish(marker)

    def publish_probs(self, probs):
        msg = Float32MultiArray()
        msg.data = probs.tolist()
        self.prob_pub.publish(msg)

    def class_idx_to_label(self, idx):
        return {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }.get(idx, 'unknown')

    def update_health_status(self, emotion):
        self.emotion_buffer.append(emotion)
        if len(self.emotion_buffer) > self.sick_threshold:
            self.emotion_buffer.pop(0)

    def is_sick(self):
        if len(self.emotion_buffer) == self.sick_threshold:
            return all(e in ['fear', 'sad'] for e in self.emotion_buffer)
        return False

    def publish_results(self, emotion):
        self.emotion_pub.publish(String(data=emotion))
        self.health_pub.publish(Bool(data=self.is_sick()))
        self.get_logger().info(f'Detected: {emotion} | Sick: {self.is_sick()}')

def main(args=None):
    rclpy.init(args=args)
    detector = EmotionDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("Shutting down...")
    finally:
        cv2.destroyAllWindows()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()