import numpy as np
import time
import cv2
import math as m
import random as rdm
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
        self.focal_length = 161
        self.last_z = 0.1
        self.last_y = 0.1
        self.state = -1
        self.lost_counter = 0 
        self.spotted_counter = 0
        ### MOVE 2 GATE VARS
        self.dst2gate = 10000
        self.gatePos = [rdm.random(), rdm.random(), rdm.random()]
        # Gate position kalman vars
        self.R = np.eye(3,3) * 0.1 
        self.Q = np.eye(3,3) * 0.3 
        self.P = np.eye(3, 3)*1000
        self.H = np.eye(3,3)

        self.currentGate = 0
        self.gateSearchTime = 0.0
        self.restPos = {0:[1.0, 2.0, 1.5, -m.pi/4],1:[4.0, 1.0, 1.5, 0.0], 2:[7.0, 2.0, 1.5, m.pi/3], 3:[7.0, 6.0, 1.5, 2*m.pi/3], 4:[4.0, 6.0, 1.5, m.pi], 5:[1.0, 4.0, 1.5, 3*m.pi/2]}
        self.nextWaypoint = [1.0, 4.0, 1.0, 0.0]
        self.gatePassed = False

        ## GATE POSITION STORED 
        self.GateWaypoints = []
    def image_filtering (self,camera_data, edge = True):
        """
        Filters camera image to suit shape detection (grayscale, cannyEdge and Shape extraction)
        
        Args : 
            BGRA numpy array
        Returns : 
            TBD
        """
        red = camera_data[:, :,2]
        green = camera_data[:, :, 1]
        score_based = red.astype(np.int16) - green
        score_based = np.clip(score_based, 0, 255)
        score_based = score_based.astype(np.uint8)
        blur = cv2.GaussianBlur(score_based,(5, 5), 0) 
        #_, score_based = cv2.threshold(score_based, 180, 220, cv2.THRESH_BINARY)
        
        edges = cv2.Canny(blur, self.Canny_threshold_low, self.Canny_threshold_high)
        countours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_dist = 100000000
        gate = np.array([])
        for cnt in countours : 
            area = cv2.contourArea(cnt)
            if area > self.MIN_AREA : 
                approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                x,y,w,h = cv2.boundingRect(approx)
                dist = self.focal_length*0.4/h
                if dist < min_dist  and len(approx) == 4 : 
                    gate = approx
        x,y, distance, angle = 0,0, 0, 0
        detect_flag = True
        if gate.size != 0 : 
            x,y, w, h = cv2.boundingRect(gate)
            x = 150 - (x + w//2)
            y = 150 - (y + h//2)
            distance = self.focal_length*0.4/h
            angle = self.findGateYaw(gate)
            camera_data = cv2.drawContours(camera_data, [gate], -1, (0, 255, 0),2)
        else : 
            detect_flag = False
        if edge : return edges, [x, y], detect_flag, distance, angle
        else : return camera_data, [x,y], detect_flag, distance, angle

    def findGateYaw(self, gate) : 
        gate = gate.reshape(4,2)
        sum = gate.sum(axis=1)
        diff = np.diff(gate, axis=1)
        top_left = gate[np.argmin(sum)]
        bottom_right = gate[np.argmax(sum)]
        top_right = gate[np.argmin(diff)]
        bottom_left = gate[np.argmax(diff)]
        # left_edge = np.linalg.norm(top_left - bottom_left)
        # right_edge = np.linalg.norm(top_right - bottom_right)
        # return ((left_edge - right_edge)/(left_edge + right_edge))*5
        mid_top    = (top_left + top_right) / 2
        mid_bottom = (bottom_left + bottom_right) / 2
        v = mid_top - mid_bottom
        return np.arctan2(v[0], v[1])  

    def filterGatePos(self, measure) : 
        Pk_k = self.P + self.Q 
        S = self.H @ Pk_k @ self.H.T + self.R
        K = Pk_k @ self.H.T @ np.linalg.inv(S)
        self.gatePos = self.gatePos + K @ (measure.T - self.H @ self.gatePos)
        print(f"gate position : {self.gatePos}")
        self.P = (np.eye(3, 3) - K@self.H) @ Pk_k
    def compute_gate_pos(self, sensor_data, gate_x_px, gate_y_px, dst2gate) : 
        rel_yaw = np.atan2(gate_x_px, self.focal_length)
        rel_pitch = np.atan2(gate_y_px, self.focal_length)
        z_rel = dst2gate*np.sin(sensor_data["pitch"] + rel_pitch)
        xy_dst = dst2gate*np.cos(sensor_data["pitch"]+ rel_pitch)
        x_rel = xy_dst*np.cos(sensor_data["yaw"] + rel_yaw)
        y_rel = xy_dst*np.sin(sensor_data["yaw"] + rel_yaw)
        print(f"xy_dst : {dst2gate}, sensor_yaw : {sensor_data["yaw"]}, rel_yaw : {rel_yaw}, x_rel : {x_rel}, y_rel {y_rel}")
        return np.array([x_rel + sensor_data["x_global"], y_rel + sensor_data["y_global"], z_rel + sensor_data["z_global"]]), rel_yaw
    def checkGateValidity(self, gate_est) -> bool : 
        dx = gate_est[0] - 4
        dy = gate_est[1] - 4 
        angle = np.atan2(dy, dx)*180/m.pi
        if angle >= (-150 + self.currentGate*60) and angle <= (-90 + self.currentGate*60) : 
            return True
        print(f"INVALID GATE. EXPECTED in between {-135 + self.currentGate*60} and {-105 + self.currentGate*60} for gate # {self.currentGate} but got {angle}")
        return False

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
        # FSM State : Finding the gate
        #TAKE OFF
        if sensor_data["z_global"] < 0.5 and self.state == -1: 
            self.camera_filtered, _, _, _, _= self.image_filtering(camera_data, False)
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return self.camera_filtered, control_command
        elif sensor_data["z_global"] >= 0.5 and self.state == -1: 
            self.state = 1
            self.camera_filtered, _, _, _, _ = self.image_filtering(camera_data, False)
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return self.camera_filtered, control_command
        if self.state == 0:
                        # #PD controller to align on gate
            # z_px = center[1]*0.01 + (center[1] - self.last_z)*dt*0.01
            # y_px = center[0]*0.005 + (center[0] - self.last_y)*dt*0.001
            # self.last_y = y_px
            # self.last_z = z_px 
            self.camera_filtered, center, flag , dst, angle2g8 = self.image_filtering(camera_data, False)
            print(f"distance to gate #{self.currentGate} is {self.dst2gate}")
            if self.dst2gate < 0.5 : 
                    self.state = 2
            if flag :
                print("GATE FOUND STATE 0")
                self.lost_counter = 0
                self.spotted_counter += 1
                gate_est, yaw_err =  self.compute_gate_pos(sensor_data, center[0], center[1], dst)
                yaw_corr, y_corr = 0, 0
                if self.checkGateValidity(gate_est) : 
                    self.filterGatePos(gate_est)
                    print(f"current estimation of the gate position : {self.gatePos}")
                    self.dst2gate = dst
                    print(angle2g8)
                    # if angle2g8 >= 0.3 : 
                    #     yaw_corr = angle2g8*0.1
                    #     y_corr = -np.sin(angle2g8)*0.5
                    # else : 
                    yaw_corr = yaw_err
                control_command = [sensor_data["x_global"] + 0.5*(self.gatePos[0] - sensor_data["x_global"]), 
                                    sensor_data["y_global"] + 0.5*(self.gatePos[1] - sensor_data["y_global"]) + y_corr,
                                    self.gatePos[2],
                                    sensor_data["yaw"] + yaw_corr] 
                return self.camera_filtered, control_command
                # print("angle to gate", angle2g8)
                # y_correction = 0
                # yaw_correction = 0
                # if np.abs(angle2g8) > 0.3 : 
                #     y_correction = angle2g8*2
                #     yaw_correction = angle2g8*0.5
            if self.spotted_counter >= 10 and not flag : 
                print("GATE NOT FOUND BUT CONFIDENT ABOUT POSITION STATE 0")
                self.dst2gate = np.linalg.norm(self.gatePos - np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]]))
                if self.dst2gate < 0.5 : 
                    self.state = 2
                self.nextWaypoint = np.array([self.gatePos[0], self.gatePos[1], self.gatePos[2], sensor_data["yaw"]])
                self.GateWaypoints.append(self.nextWaypoint)
                control_command = [sensor_data["x_global"] + 0.5*(self.gatePos[0] - sensor_data["x_global"]), 
                                    sensor_data["y_global"] + 0.5*(self.gatePos[1] - sensor_data["y_global"]),
                                    self.gatePos[2] ,
                                    sensor_data["yaw"] + angle2g8] 
                return self.camera_filtered, control_command
            # control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            # if self.camera_filtered is not None : 
            if self.spotted_counter < 10 and not flag : 
                print("GATE NOT FOUND AND NOT CONFIDENT ON POSITION")
                self.lost_counter += dt
                if self.lost_counter > 2 : 
                    self.state = 1
                    control_command = [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"], sensor_data["yaw"]]
                    return self.camera_filtered, control_command
                else : 
                    control_command = [sensor_data["x_global"] + 0.5*(self.gatePos[0] - sensor_data["x_global"]), 
                                   sensor_data["y_global"] + 0.5*(self.gatePos[1] - sensor_data["y_global"]),
                                   sensor_data["z_global"] + 0.5*(self.gatePos[2] - sensor_data["z_global"]),
                                   sensor_data["yaw"]] 
                    return self.camera_filtered, control_command

        elif self.state == 1 : 
            self.camera_filtered, center, flag , dst, _ = self.image_filtering(camera_data)
            y_pos  = self.restPos[self.currentGate][1] 
            if flag: 
                gate_est, _ = self.compute_gate_pos(sensor_data, center[0], center[1], dst)
                if self.checkGateValidity(gate_est) : 
                    self.state = 0
                self.gateSearchTime += dt
                control_command = [sensor_data["x_global"], y_pos , 1.5 + 0.5*np.sin(self.gateSearchTime/6), m.pi/3*np.sin(self.gateSearchTime/3) + self.currentGate*m.pi/3 - m.pi/3]
                return self.camera_filtered, control_command
            else : 
                if sensor_data["z_global"] < 0.9 : 
                    control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, -m.pi/4]
                else : 
                    self.gateSearchTime += dt
                    print(self.gateSearchTime) 
                    control_command = [sensor_data["x_global"], y_pos , 1.5 + 0.5*np.sin(self.gateSearchTime/6), m.pi/3*np.sin(self.gateSearchTime/3) + self.currentGate*m.pi/3 - m.pi/12]
            return self.camera_filtered, control_command
        elif self.state == 2: 
            self.camera_filtered, center, flag, dst, _ = self.image_filtering(camera_data, False)
            current_pos = np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]])
            pos_err = self.nextWaypoint[:3] - current_pos
            if np.linalg.norm(pos_err) < 0.1 :
                if not self.gatePassed : 
                    self.currentGate += 1
                    self.nextWaypoint = self.restPos[self.currentGate]
                    self.gatePassed = True
                    self.P = np.eye(3,3)*100
                    self.dst2gate = 100000 
                    self.spotted_counter = 0
                else : 
                    self.gatePassed = False
                    self.state = 1
            elif self.gatePassed : 
                if flag : 
                    gate_est, yaw_err = self.compute_gate_pos(sensor_data, center[0], center[1], dst)
                    self.filterGatePos(gate_est)
                    self.spotted_counter += 1 
                    if self.spotted_counter >= 10 : 
                        self.state = 0
                        self.gatePassed = False
            control_command = [sensor_data["x_global"] + 0.5*(self.nextWaypoint[0] - sensor_data["x_global"]), 
                                   sensor_data["y_global"] + 0.5*(self.nextWaypoint[1] - sensor_data["y_global"]),
                                   sensor_data["z_global"] + 0.5*(self.nextWaypoint[2] - sensor_data["z_global"]),
                                   self.nextWaypoint[3]]
            return self.camera_filtered, control_command       
             

        
        # else : 
        #     return None
        # return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
        # Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    image, command = _controller.compute_command(sensor_data, camera_data, dt)
    if command[2] < 0.5 : 
        command[2] = 0.5
    return image, command

