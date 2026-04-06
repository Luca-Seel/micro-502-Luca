import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


class MyAssignment:
    def __init__(self):
        # ---------- VARs --------------
        # CV vars 
        self.MIN_AREA = 500
        self.camera_filtered = None
        self.Canny_threshold_low = 75
        self.Canny_threshold_high = 150

    def image_filtering (self,camera_data):
        """
        Filters camera image to suit shape detection (grayscale, cannyEdge and Shape extraction)
        
        Args : 
            BGRA numpy array
        Returns : 
            TBD
        """
        red = camera_data[:, :,2]
        _, red = cv2.threshold(red, 180, 220, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(red,(5, 5), 0) 
        edges = cv2.Canny(blur, self.Canny_threshold_low, self.Canny_threshold_high)
        countours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        gate = np.array([])
        for cnt in countours : 
            area = cv2.contourArea(cnt)
            if area > self.MIN_AREA : 
                approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                if area > max_area and len(approx) == 4 : 
                    gate = approx
        cv2.drawContours(camera_data, [gate], -1, (0, 255, 0),2) 
        return [edges, camera_data]

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        # if sensor_data['z_global'] < 0.49:
        #     control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        #     return control_command

        # ---- YOUR CODE HERE ----
        """
        TODO : 
        - Computer Vision to find the gates
        - Extract inertial frame position of gates
        - Extract setpoints
        - Compute motor commands
        """
        # ------- COMPUTER VISION ------- 
        self.camera_filtered = self.image_filtering(camera_data)
        
        # control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        # if self.camera_filtered is not None : 
        return self.camera_filtered
        # else : 
        #     return None
        # return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
