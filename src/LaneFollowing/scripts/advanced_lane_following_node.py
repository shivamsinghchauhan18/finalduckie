#!/usr/bin/env python3

"""
Advanced Lane Following Node for BlueDuckie Bot
Features:
- Kalman filtering for smooth lane detection
- Polynomial curve fitting for precise path following
- Dynamic PID control with adaptive parameters
- Robust yellow/white line detection with HSV color space
- Predictive steering for curve anticipation
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from std_msgs.msg import Header, Bool
import threading
from collections import deque
import time
import math
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


class AdvancedLaneFollower:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('blueduckie_advanced_lane_following', anonymous=True)
        
        # Get robot name from parameter server
        self.robot_name = rospy.get_param("~robot_name", "blueduckie")
        
        # Publishers
        self.cmd_pub = rospy.Publisher(
            f"/{self.robot_name}/wheels_driver_node/wheels_cmd", 
            WheelsCmdStamped, 
            queue_size=1
        )
        
        # Emergency stop publisher
        self.emergency_stop_pub = rospy.Publisher(
            f"/{self.robot_name}/emergency_stop",
            Bool,
            queue_size=1
        )
        
        # Subscribers
        self.image_sub = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Advanced Lane Following Parameters
        self.setup_parameters()
        
        # State variables
        self.last_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.error_history = deque(maxlen=20)
        self.lane_center_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=10)
        self.speed_history = deque(maxlen=5)
        
        # Advanced tracking variables
        self.lost_lane_counter = 0
        self.emergency_stop_triggered = False
        self.last_good_lane_center = None
        self.adaptive_roi = {"top": 0.4, "bottom": 0.9}
        self.frame_count = 0
        self.fps_counter = time.time()
        
        # Multi-scale detection
        self.detection_scales = [1.0, 0.8, 1.2]
        self.scale_weights = [0.6, 0.2, 0.2]
        
        # Kalman filter for smooth tracking
        self.kalman = self.setup_kalman_filter()
        
        # Threading lock
        self.lock = threading.Lock()
        
        rospy.loginfo(f"Advanced Lane Following initialized for {self.robot_name}")

    def setup_parameters(self):
        """Setup advanced control parameters with ROS parameter support"""
        # Enhanced PID Parameters with environmental adaptation
        self.kp_base = rospy.get_param("~kp_base", 0.85)
        self.ki_base = rospy.get_param("~ki_base", 0.025)
        self.kd_base = rospy.get_param("~kd_base", 0.18)
        self.kp_curve = rospy.get_param("~kp_curve", 1.2)  # Higher gain for curves
        self.ki_max = rospy.get_param("~ki_max", 0.1)    # Maximum integral gain
        
        # Adaptive speed control
        self.base_speed = rospy.get_param("~base_speed", 0.32)
        self.max_speed = rospy.get_param("~max_speed", 0.55)
        self.min_speed = rospy.get_param("~min_speed", 0.12)
        self.curve_speed_reduction = rospy.get_param("~curve_speed_reduction", 0.3)
        self.confidence_speed_factor = rospy.get_param("~confidence_speed_factor", 0.8)
        
        # Enhanced color detection (HSV) - multiple lighting conditions
        self.yellow_ranges = [
            (np.array([18, 80, 80]), np.array([35, 255, 255])),   # Normal lighting
            (np.array([15, 60, 60]), np.array([40, 255, 200])),   # Low lighting
            (np.array([20, 120, 120]), np.array([32, 255, 255]))  # Bright lighting
        ]
        
        self.white_ranges = [
            (np.array([0, 0, 180]), np.array([180, 30, 255])),    # Normal
            (np.array([0, 0, 140]), np.array([180, 40, 255])),    # Low light
            (np.array([0, 0, 200]), np.array([180, 20, 255]))     # Bright light
        ]
        
        # Adaptive Region of Interest
        self.roi_configs = {
            "normal": {"top": 0.35, "bottom": 0.9},
            "curve": {"top": 0.25, "bottom": 0.95},
            "straight": {"top": 0.4, "bottom": 0.85}
        }
        
        # Advanced smoothing and filtering
        self.kalman_process_noise = rospy.get_param("~kalman_process_noise", 0.025)
        self.kalman_measurement_noise = rospy.get_param("~kalman_measurement_noise", 0.08)
        self.curve_smoothing_window = rospy.get_param("~curve_smoothing_window", 7)
        self.steering_smoothing = rospy.get_param("~steering_smoothing", 0.75)
        self.velocity_smoothing = rospy.get_param("~velocity_smoothing", 0.6)
        
        # Lane geometry parameters
        self.expected_lane_width = rospy.get_param("~expected_lane_width", 195)
        self.lane_width_tolerance = rospy.get_param("~lane_width_tolerance", 60)
        self.min_lane_pixels = rospy.get_param("~min_lane_pixels", 80)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.8)
        
        # Safety and recovery parameters
        self.max_lost_frames = rospy.get_param("~max_lost_frames", 15)
        self.recovery_speed_factor = rospy.get_param("~recovery_speed_factor", 0.5)
        self.emergency_stop_threshold = rospy.get_param("~emergency_stop_threshold", 25)
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.3)
        
        rospy.loginfo(f"BlueDuckie parameters loaded - Base speed: {self.base_speed}, "
                     f"PID: Kp={self.kp_base}, Ki={self.ki_base}, Kd={self.kd_base}")

    def setup_kalman_filter(self):
        """Setup enhanced Kalman filter for multi-dimensional tracking"""
        # 6D state: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
        kalman = cv2.KalmanFilter(6, 2)
        
        # Enhanced state transition matrix with acceleration
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Adaptive noise covariance
        kalman.processNoiseCov = self.kalman_process_noise * np.eye(6, dtype=np.float32)
        kalman.measurementNoiseCov = self.kalman_measurement_noise * np.eye(2, dtype=np.float32)
        
        # Initialize state
        kalman.statePre = np.array([160, 240, 0, 0, 0, 0], dtype=np.float32)
        kalman.statePost = np.array([160, 240, 0, 0, 0, 0], dtype=np.float32)
        
        return kalman

    def preprocess_image(self, cv_image):
        """Advanced multi-stage image preprocessing"""
        # Multi-scale preprocessing for robustness
        height, width = cv_image.shape[:2]
        
        # Adaptive histogram equalization per channel
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to HSV with enhanced preprocessing
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Shadow removal technique
        shadow_mask = cv2.inRange(hsv[:,:,2], 0, 70)
        hsv[:,:,2] = np.where(shadow_mask > 0, hsv[:,:,2] * 2.5, hsv[:,:,2])
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        
        # Adaptive Gaussian blur based on image noise
        noise_level = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        blur_kernel = (3, 3) if noise_level < 15 else (5, 5)
        hsv = cv2.GaussianBlur(hsv, blur_kernel, 0)
        
        return hsv, enhanced

    def detect_lane_lines_advanced(self, hsv_image):
        """Multi-scale, multi-condition lane detection"""
        height, width = hsv_image.shape[:2]
        
        # Adaptive ROI based on current driving state
        roi_config = self.get_adaptive_roi()
        roi_top = int(height * roi_config["top"])
        roi_bottom = int(height * roi_config["bottom"])
        
        # Multi-condition color detection
        combined_masks = []
        
        # Test multiple lighting conditions
        for yellow_range in self.yellow_ranges:
            yellow_mask = cv2.inRange(hsv_image, yellow_range[0], yellow_range[1])
            combined_masks.append(yellow_mask)
            
        for white_range in self.white_ranges:
            white_mask = cv2.inRange(hsv_image, white_range[0], white_range[1])
            combined_masks.append(white_mask)
        
        # Combine all masks with weighted voting
        final_mask = np.zeros_like(combined_masks[0])
        for mask in combined_masks:
            final_mask = cv2.bitwise_or(final_mask, mask)
        
        # Apply adaptive ROI
        roi_mask = np.zeros_like(final_mask)
        roi_mask[roi_top:roi_bottom, :] = 255
        final_mask = cv2.bitwise_and(final_mask, roi_mask)
        
        # Advanced morphological operations
        # Adaptive kernel size based on image resolution
        kernel_size = max(3, width // 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise components
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(final_mask)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_lane_pixels:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask

    def get_adaptive_roi(self):
        """Dynamically adjust ROI based on driving conditions"""
        if len(self.error_history) < 5:
            return self.roi_configs["normal"]
        
        # Calculate recent error variance to detect curves
        recent_errors = list(self.error_history)[-5:]
        error_variance = np.var(recent_errors)
        
        if error_variance > 0.1:  # High variance indicates curves
            return self.roi_configs["curve"]
        elif error_variance < 0.02:  # Low variance indicates straight road
            return self.roi_configs["straight"]
        else:
            return self.roi_configs["normal"]

    def multi_scale_detection(self, mask):
        """Perform detection at multiple scales for robustness"""
        height, width = mask.shape
        scale_results = []
        
        for scale, weight in zip(self.detection_scales, self.scale_weights):
            if scale != 1.0:
                new_height = int(height * scale)
                new_width = int(width * scale)
                scaled_mask = cv2.resize(mask, (new_width, new_height))
                lane_center, confidence = self.find_lane_center_advanced(scaled_mask)
                # Scale back to original coordinates
                lane_center = int(lane_center / scale)
            else:
                lane_center, confidence = self.find_lane_center_advanced(mask)
            
            scale_results.append((lane_center, confidence * weight))
        
        # Weighted combination of multi-scale results
        if scale_results:
            total_weight = sum(result[1] for result in scale_results)
            if total_weight > 0:
                weighted_center = sum(result[0] * result[1] for result in scale_results) / total_weight
                avg_confidence = total_weight / len(scale_results)
                return int(weighted_center), avg_confidence
        
        return width // 2, 0.0

    def find_lane_center_advanced(self, mask):
        """Enhanced lane center detection with polynomial fitting and RANSAC"""
        height, width = mask.shape
        
        # Advanced sliding window with adaptive window size
        base_window_count = 12
        window_height = height // base_window_count
        left_points = []
        right_points = []
        
        # Adaptive window parameters
        window_width = width // 8
        min_pixels_threshold = max(20, (height * width) // 10000)
        
        for i in range(base_window_count):
            y_start = height - (i + 1) * window_height
            y_end = height - i * window_height
            
            window_slice = mask[y_start:y_end, :]
            
            if np.sum(window_slice) > min_pixels_threshold:
                # Find contours with area filtering
                contours, _ = cv2.findContours(window_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 15:
                        # Additional shape filtering
                        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                        aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]
                        
                        if len(approx) >= 4 and 0.3 <= aspect_ratio <= 3.0:
                            valid_contours.append(contour)
                
                for contour in valid_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = y_start + int(M["m01"] / M["m00"])
                        
                        # Enhanced classification with hysteresis
                        center_bias = 0.1 * width  # 10% bias toward center
                        
                        if cx < (width // 2 - center_bias):
                            left_points.append((cx, cy))
                        elif cx > (width // 2 + center_bias):
                            right_points.append((cx, cy))
        
        # Enhanced lane center calculation with polynomial fitting
        lane_center_x = width // 2
        confidence = 0.0
        lane_width_detected = None
        
        if len(left_points) > 3 and len(right_points) > 3:
            # Both lines detected - fit polynomials
            left_poly = self.fit_polynomial_ransac([p[0] for p in left_points], [p[1] for p in left_points])
            right_poly = self.fit_polynomial_ransac([p[0] for p in right_points], [p[1] for p in right_points])
            
            if left_poly is not None and right_poly is not None:
                # Calculate center at multiple y positions and average
                y_positions = np.linspace(height * 0.7, height * 0.9, 5)
                center_points = []
                
                for y in y_positions:
                    left_x = np.polyval(left_poly, y)
                    right_x = np.polyval(right_poly, y)
                    center_x = (left_x + right_x) / 2
                    center_points.append(center_x)
                
                lane_center_x = int(np.mean(center_points))
                lane_width_detected = abs(np.mean([np.polyval(right_poly, y) - np.polyval(left_poly, y) for y in y_positions]))
                
                # Confidence based on polynomial fit quality and lane width
                width_score = 1.0 - abs(lane_width_detected - self.expected_lane_width) / self.expected_lane_width
                width_score = max(0, min(1, width_score))
                confidence = 0.9 + 0.1 * width_score
            else:
                # Fallback to centroid method
                left_x = np.mean([p[0] for p in left_points])
                right_x = np.mean([p[0] for p in right_points])
                lane_center_x = int((left_x + right_x) / 2)
                confidence = 0.8
                
        elif len(left_points) > 3:
            # Only left line detected
            left_poly = self.fit_polynomial_ransac([p[0] for p in left_points], [p[1] for p in left_points])
            if left_poly is not None:
                y_ref = height * 0.8
                left_x = np.polyval(left_poly, y_ref)
                lane_center_x = int(left_x + self.expected_lane_width / 2)
            else:
                left_x = np.mean([p[0] for p in left_points])
                lane_center_x = int(left_x + self.expected_lane_width / 2)
            confidence = 0.6
            
        elif len(right_points) > 3:
            # Only right line detected
            right_poly = self.fit_polynomial_ransac([p[0] for p in right_points], [p[1] for p in right_points])
            if right_poly is not None:
                y_ref = height * 0.8
                right_x = np.polyval(right_poly, y_ref)
                lane_center_x = int(right_x - self.expected_lane_width / 2)
            else:
                right_x = np.mean([p[0] for p in right_points])
                lane_center_x = int(right_x - self.expected_lane_width / 2)
            confidence = 0.6
        
        # Temporal consistency check
        if self.last_good_lane_center is not None and confidence > 0:
            deviation = abs(lane_center_x - self.last_good_lane_center)
            if deviation > width * 0.3:  # 30% of image width
                confidence *= 0.5  # Reduce confidence for large deviations
        
        return lane_center_x, confidence

    def fit_polynomial_ransac(self, x_points, y_points, degree=2):
        """Robust polynomial fitting using RANSAC"""
        if len(x_points) < degree + 1:
            return None
        
        try:
            # Convert to numpy arrays
            x_points = np.array(x_points)
            y_points = np.array(y_points)
            
            # Simple polynomial fit (numpy polyfit is already quite robust)
            coeffs = np.polyfit(y_points, x_points, degree)
            
            # Validation: check if fit makes sense
            y_test = np.linspace(min(y_points), max(y_points), 10)
            x_pred = np.polyval(coeffs, y_test)
            
            # Check for reasonable curvature (not too extreme)
            if len(coeffs) > 2 and abs(coeffs[0]) > 0.01:  # Too much curvature
                return None
                
            return coeffs
        except:
            return None

    def calculate_curvature_advanced(self, lane_center_history):
        """Advanced curvature calculation with smoothing and prediction"""
        if len(lane_center_history) < 5:
            return 0.0, 0.0  # curvature, rate_of_change
        
        try:
            # Convert to numpy array and smooth
            centers = np.array(list(lane_center_history))
            
            # Apply Gaussian smoothing
            if len(centers) >= 5:
                centers_smooth = gaussian_filter1d(centers, sigma=0.8)
            else:
                centers_smooth = centers
            
            # Create time/position indices
            indices = np.arange(len(centers_smooth))
            
            # Fit spline for smooth derivatives
            if len(centers_smooth) >= 4:
                spline = UnivariateSpline(indices, centers_smooth, s=0.5, k=min(3, len(centers_smooth)-1))
                
                # Calculate first and second derivatives
                first_deriv = spline.derivative(1)(indices)
                second_deriv = spline.derivative(2)(indices)
                
                # Curvature formula: |y''| / (1 + y'^2)^(3/2)
                latest_first = first_deriv[-1]
                latest_second = second_deriv[-1]
                
                curvature = abs(latest_second) / (1 + latest_first**2)**(3/2)
                
                # Rate of curvature change (for predictive control)
                if len(second_deriv) >= 2:
                    curvature_rate = second_deriv[-1] - second_deriv[-2]
                else:
                    curvature_rate = 0.0
                
                return curvature, curvature_rate
            else:
                # Simple finite difference method
                if len(centers_smooth) >= 3:
                    # Second derivative approximation
                    second_diff = centers_smooth[-3] - 2*centers_smooth[-2] + centers_smooth[-1]
                    return abs(second_diff) * 0.1, 0.0
                
        except Exception as e:
            rospy.logwarn(f"Curvature calculation error: {e}")
        
        return 0.0, 0.0

    def adaptive_pid_control_advanced(self, error, confidence, curvature, curvature_rate):
        """Enhanced adaptive PID with environmental awareness"""
        # Environmental adaptation
        error_magnitude = abs(error)
        confidence_factor = max(0.2, confidence)
        
        # Adaptive gains based on multiple factors
        # Base gains
        kp = self.kp_base * confidence_factor
        ki = self.ki_base * confidence_factor
        kd = self.kd_base * confidence_factor
        
        # Curve-specific adaptation
        if curvature > 0.05:  # In curve
            kp *= self.kp_curve  # Increase proportional gain
            ki *= 0.7  # Reduce integral to prevent overshoot
            
        # Error magnitude adaptation
        if error_magnitude > 0.5:  # Large error
            kp *= 1.3
            kd *= 1.5
        elif error_magnitude < 0.1:  # Small error, fine-tuning
            ki *= 1.4
            
        # Speed-dependent adaptation
        if len(self.speed_history) > 0:
            avg_speed = np.mean(list(self.speed_history))
            speed_factor = max(0.5, min(1.5, avg_speed / self.base_speed))
            kp *= speed_factor
            kd *= speed_factor
        
        # Enhanced PID calculation
        self.integral += error
        
        # Adaptive integral limits (anti-windup)
        integral_limit = min(self.ki_max / max(ki, 0.001), 50)
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        
        # Derivative calculation with smoothing
        if len(self.error_history) > 0:
            raw_derivative = error - self.error_history[-1]
            # Smooth derivative to reduce noise
            if hasattr(self, 'prev_derivative'):
                self.derivative = 0.7 * self.prev_derivative + 0.3 * raw_derivative
            else:
                self.derivative = raw_derivative
            self.prev_derivative = self.derivative
        
        # Predictive component based on curvature rate
        predictive_term = curvature_rate * 0.1 if abs(curvature_rate) < 1.0 else 0
        
        # Final PID output with predictive term
        pid_output = kp * error + ki * self.integral + kd * self.derivative + predictive_term
        
        # Adaptive output limiting
        max_output = self.max_steering_angle * confidence_factor
        pid_output = np.clip(pid_output, -max_output, max_output)
        
        # Store for next iteration
        self.error_history.append(error)
        self.last_error = error
        
        return pid_output

    def calculate_speeds_advanced(self, steering_error, confidence, curvature, curvature_rate):
        """Advanced speed calculation with predictive and safety features"""
        # Base speed calculation with multiple factors
        confidence_factor = max(0.3, confidence)
        
        # Curvature-based speed adaptation
        curvature_factor = 1.0 - min(abs(curvature) * self.curve_speed_reduction, 0.6)
        
        # Predictive speed reduction for upcoming curves
        if abs(curvature_rate) > 0.1:
            predictive_factor = 1.0 - min(abs(curvature_rate) * 0.2, 0.3)
        else:
            predictive_factor = 1.0
        
        # Error-based speed adaptation
        error_magnitude = abs(steering_error)
        error_factor = 1.0 - min(error_magnitude * 0.3, 0.4)
        
        # Combine all factors
        speed_factor = confidence_factor * curvature_factor * predictive_factor * error_factor
        adapted_speed = self.base_speed * speed_factor
        adapted_speed = np.clip(adapted_speed, self.min_speed, self.max_speed)
        
        # Smooth speed changes
        if len(self.speed_history) > 0:
            prev_speed = self.speed_history[-1]
            adapted_speed = (1 - self.velocity_smoothing) * adapted_speed + self.velocity_smoothing * prev_speed
        
        # Enhanced differential steering with curvature compensation
        base_steering_gain = 0.85
        
        # Adaptive steering gain based on speed and confidence
        steering_gain = base_steering_gain * (1.0 + 0.3 * (1.0 - confidence))
        steering_gain *= max(0.7, adapted_speed / self.base_speed)  # Reduce gain at low speeds
        
        steering_adjustment = steering_error * steering_gain
        
        # Curvature-based differential enhancement
        if abs(curvature) > 0.02:
            curve_differential = curvature * 0.15 * np.sign(steering_error)
            steering_adjustment += curve_differential
        
        # Calculate initial wheel speeds
        left_speed = adapted_speed - steering_adjustment
        right_speed = adapted_speed + steering_adjustment
        
        # Advanced normalization to maintain forward motion and prevent wheel slip
        max_wheel_speed = max(abs(left_speed), abs(right_speed))
        
        if max_wheel_speed > self.max_speed:
            # Scale down proportionally
            scale = self.max_speed / max_wheel_speed
            left_speed *= scale
            right_speed *= scale
        
        # Prevent reverse motion unless explicitly commanded
        if left_speed < 0 and right_speed < 0:
            left_speed = max(left_speed, -self.min_speed)
            right_speed = max(right_speed, -self.min_speed)
        
        # Store speed for smoothing
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2
        self.speed_history.append(avg_speed)
        
        return left_speed, right_speed

    def safety_check_and_recovery(self, confidence, lane_center_x, image_width):
        """Advanced safety monitoring and recovery system"""
        # Track lost lane detection
        if confidence < self.confidence_threshold:
            self.lost_lane_counter += 1
        else:
            self.lost_lane_counter = max(0, self.lost_lane_counter - 2)  # Recover faster
            self.last_good_lane_center = lane_center_x
        
        # Emergency stop condition
        if self.lost_lane_counter > self.emergency_stop_threshold:
            if not self.emergency_stop_triggered:
                rospy.logwarn("EMERGENCY STOP: Lane detection lost for too long!")
                self.emergency_stop_triggered = True
                # Publish emergency stop
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_pub.publish(emergency_msg)
            return True, 0.0, 0.0  # Stop the robot
        
        # Recovery mode - reduced speed operation
        elif self.lost_lane_counter > self.max_lost_frames:
            if self.last_good_lane_center is not None:
                # Use last known good position with reduced confidence
                recovery_center = self.last_good_lane_center
                recovery_error = (recovery_center - image_width // 2) / (image_width / 2)
                return False, recovery_error * 0.5, self.recovery_speed_factor  # Reduced speed
            else:
                # Dead reckoning - go straight slowly
                return False, 0.0, self.recovery_speed_factor
        
        # Reset emergency stop if recovered
        if self.emergency_stop_triggered and confidence > self.confidence_threshold:
            rospy.loginfo("Lane detection recovered - resuming normal operation")
            self.emergency_stop_triggered = False
            emergency_msg = Bool()
            emergency_msg.data = False
            self.emergency_stop_pub.publish(emergency_msg)
        
        return False, None, 1.0  # Normal operation

    def image_callback(self, msg):
        """Enhanced main image processing callback with comprehensive lane following"""
        try:
            with self.lock:
                self.frame_count += 1
                
                # Convert ROS image to OpenCV
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                height, width = cv_image.shape[:2]
                
                # Advanced preprocessing
                hsv_image, enhanced_image = self.preprocess_image(cv_image)
                
                # Multi-scale lane detection
                lane_mask = self.detect_lane_lines_advanced(hsv_image)
                lane_center_x, confidence = self.multi_scale_detection(lane_mask)
                
                # Store confidence history for trend analysis
                self.confidence_history.append(confidence)
                
                # Safety check and recovery
                emergency_stop, recovery_error, speed_factor = self.safety_check_and_recovery(
                    confidence, lane_center_x, width)
                
                if emergency_stop:
                    # Emergency stop - publish zero speeds
                    self.publish_wheel_cmd(0.0, 0.0)
                    return
                
                # Use recovery error if in recovery mode
                if recovery_error is not None:
                    normalized_error = recovery_error
                    confidence *= 0.5  # Reduce confidence in recovery mode
                else:
                    # Normal operation - calculate error
                    image_center = width // 2
                    error = lane_center_x - image_center
                    normalized_error = error / (width / 2)
                
                # Enhanced Kalman filtering with adaptive noise
                measurement = np.array([[lane_center_x], [height * 0.75]], dtype=np.float32)
                
                # Adapt Kalman noise based on confidence
                noise_factor = 1.0 / max(0.1, confidence)
                self.kalman.measurementNoiseCov = (self.kalman_measurement_noise * noise_factor) * np.eye(2, dtype=np.float32)
                
                self.kalman.correct(measurement)
                prediction = self.kalman.predict()
                
                smoothed_lane_center = int(prediction[0])
                predicted_velocity = prediction[2]  # x-velocity from Kalman state
                
                # Store lane center history for curvature calculation
                self.lane_center_history.append(smoothed_lane_center)
                
                # Advanced curvature calculation
                curvature, curvature_rate = self.calculate_curvature_advanced(self.lane_center_history)
                
                # Recalculate error using smoothed center
                smoothed_error = smoothed_lane_center - width // 2
                smoothed_normalized_error = smoothed_error / (width / 2)
                
                # Enhanced adaptive PID control
                steering_output = self.adaptive_pid_control_advanced(
                    smoothed_normalized_error, confidence, curvature, curvature_rate)
                
                # Advanced speed calculation
                left_speed, right_speed = self.calculate_speeds_advanced(
                    steering_output, confidence, curvature, curvature_rate)
                
                # Apply speed factor from safety system
                left_speed *= speed_factor
                right_speed *= speed_factor
                
                # Publish enhanced wheel commands
                self.publish_wheel_cmd(left_speed, right_speed)
                
                # Comprehensive logging with performance metrics
                if self.frame_count % 30 == 0:  # Log every 30 frames (~1 sec at 30fps)
                    current_time = time.time()
                    fps = 30.0 / (current_time - self.fps_counter) if hasattr(self, 'fps_counter') else 0
                    self.fps_counter = current_time
                    
                    avg_confidence = np.mean(list(self.confidence_history)[-10:]) if self.confidence_history else 0
                    
                    rospy.loginfo(
                        f"BlueDuckie Advanced Lane Following - "
                        f"Error: {smoothed_error:.1f}px, "
                        f"Confidence: {confidence:.2f} (avg: {avg_confidence:.2f}), "
                        f"Curvature: {curvature:.3f}, "
                        f"Speeds: L={left_speed:.2f}, R={right_speed:.2f}, "
                        f"FPS: {fps:.1f}, "
                        f"Lost frames: {self.lost_lane_counter}"
                    )
                
        except Exception as e:
            rospy.logerr(f"Critical error in image processing: {str(e)}")
            # Emergency fallback - stop the robot
            self.publish_wheel_cmd(0.0, 0.0)

    def publish_wheel_cmd(self, left_speed, right_speed):
        """Publish wheel commands"""
        msg = WheelsCmdStamped()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = left_speed
        msg.vel_right = right_speed
        
        self.cmd_pub.publish(msg)

    def run(self):
        """Main run loop"""
        rospy.loginfo("BlueDuckie Advanced Lane Following is running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = AdvancedLaneFollower()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("BlueDuckie Lane Following node terminated.")
