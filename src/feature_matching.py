from collections import Counter, defaultdict
import cv2
import json
import math
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import sys
import torch
from ultralytics import YOLO
import zmq


#import SuperGlue from local
superglue_path = "/home/ipanagias/Documents/aria_pkg/src/SuperGluePretrainedNetwork/"
if superglue_path not in sys.path:
    sys.path.append(superglue_path)
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (make_matching_plot, AverageTimer, read_image, frame2tensor)


# disable gradient calculation globally
torch.set_grad_enabled(False)



def superglue_matching_init():
    device = 'cuda' 
    config = {
        'superpoint': {
            'nms_radius': 2,
            'keypoint_threshold': 0.0,
            'max_keypoints': 10000,
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.00001,
        }
    }
    matching = Matching(config).eval().to(device)
    timer = AverageTimer(newline=True)
    return matching


def superglue(matching, aria_img, robo_img, bboxesaria, filepath):
    

    # compute tensor of images
    device = "cuda"
    aria_inp = frame2tensor(aria_img, device)
    robo_inp = frame2tensor(robo_img, device)

    # match both pictures and convert to numpy arrays
    pred = matching({'image0': aria_inp, 'image1': robo_inp})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    ariakpts, robokpts = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    print("Raw matches:", np.sum(matches > -1))

    aria_matched_list = []
    robo_matched_list = []
    mconf_list = []
    n = 0
    for bboxaria in bboxesaria:
        image_name = "superglue/matchresult" + str(n) + ".png"
        viz_path = os.path.join(filepath, image_name)
        # write matches to dictionary and filter by bbox
        out_matches = {'keypoints0': ariakpts, 'keypoints1': robokpts,
                       'matches': matches, 'match_confidence': conf}
        filtered_ariakpts, filtered_matches, filtered_conf, valid_bbox = filter_points_by_bbox(ariakpts, matches, conf, bboxaria)
        
        # keep valid matched keypoints
        valid = filtered_matches > -1
        aria_matched = filtered_ariakpts[valid]
        robo_matched = robokpts[filtered_matches[valid]]
        mconf = filtered_conf[valid & valid_bbox]
        # visualize matches
        color = cm.jet(mconf)
        text = []
        make_matching_plot(
            aria_img, robo_img, ariakpts, robokpts,
            aria_matched, robo_matched,
            color, text, viz_path)
        n += 1
        aria_matched_list.append(aria_matched)
        robo_matched_list.append(robo_matched)
        mconf_list.append(mconf)
    return aria_matched_list, robo_matched_list, mconf_list


def filter_points_by_bbox(keypoints, matches, conf, bbox):
    x_min, y_min, x_max, y_max = bbox
    tolerance = 30
    x_min -= tolerance
    y_min -= tolerance
    x_max += tolerance
    y_max += tolerance

    valid = (keypoints[:, 0] >= x_min) & (keypoints[:, 0] <= x_max) & \
            (keypoints[:, 1] >= y_min) & (keypoints[:, 1] <= y_max)
    print(f"Valid keypoints in bbox: {np.sum(valid)}")
    print(f"Valid matches in bbox: {np.sum(matches[valid] > -1)}")
    return keypoints[valid], matches[valid], conf[valid], valid[valid]


def calculate_matching_points_in_box(mkpts1, boxes):
    points_per_bbox = []
    tolerance = 15 
    for box in boxes:
        bbox_tensor = box.xyxy[0]  # bbox coordinates as tensor
        x_min, y_min, x_max, y_max = bbox_tensor.tolist()
        # check which mkpt is in box
        valid = (mkpts1[:, 0] >= x_min-tolerance) & (mkpts1[:, 0] <= x_max+tolerance) & \
                (mkpts1[:, 1] >= y_min-tolerance) & (mkpts1[:, 1] <= y_max+tolerance)
        # calculate sum
        points_per_bbox.append(np.sum(valid))

    max_bbox_index = np.argmax(points_per_bbox)
    return max_bbox_index, points_per_bbox


def get_robo_img_bbox(context, maskModel, ip):
    socket = context.socket(zmq.REQ)
    socket.connect(ip + ":5562")
    print("ZMQ socket connected to Panda3 PC")

    # request image from IntelRealSense camera (roboter)
    socket.send(b"send_image")
    print("Request sent to Panda3 PC for image...")

    # receive image in bytes, decode to NumPy array and convert to grayscale
    img_bytes = socket.recv()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR format
    if img is not None:
        print(f"Received image from robot")
    rot_img = np.rot90(img, -2)
    gray_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)

    # get bounding boxes from YOLO
    registered_bricks = get_brick_poses(rot_img, maskModel)
    #annotated_frame = registered_bricks.plot()  

    # # visualize images if needed
    #cv2.imshow("Robo Image", rot_img)
    #cv2.imshow("Gray Image", gray_img)
    #cv2.imshow("BBoxes", annotated_frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if registered_bricks is not None:
        socket.close()
        return gray_img, registered_bricks.boxes
    
    socket.close()
    return gray_img, None
    '''
    for brick in registered_bricks.boxes:
        bbox_tensor = brick.xyxy[0]  # bbox coordinates as tensor
        x1, y1, x2, y2 = bbox_tensor.tolist()
        bboxesrobo.append([x1, y1, x2, y2])
    socket.close()
    return gray_img, registered_bricks.boxes
    '''


def get_aria_img_bbox(context, maskModel, nlp_json, filepath, ip):

    et_csv_file_path = os.path.join(filepath, "eyetracking/general_eye_gaze.csv")
    rgb_csv_file_path = os.path.join(filepath, "rgbcam/data.csv")
    rgb_path = os.path.join(filepath, "undistorted_imgs/")

    nlp_string = json.dumps(nlp_json)
    nlp_command = nlp_json["function_name"][0]
    
    #check if feature matching is needed or different function is called
    if nlp_command == "grab_brick" or nlp_command == "place_brick":
        intent_json = nlp_json["arguments"][0]
    else:
        #sys.exit() # remove when robot set up
        publish_bbox(context, nlp_string, None, None, ip)
        return None
    
    # get first timestamp to extract context img
    context_frame_timestamp = intent_json.get('startTime_ns', [])[0]
    filename = extract_timestamp(rgb_csv_file_path, [context_frame_timestamp])

    # open RGB img from first timestamp and convert to grayscale
    rgb_file = rgb_path+filename
    rgb_img = np.load(rgb_file)
    rot_img = np.rot90(rgb_img, -1)
    color_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    registered_bricks = get_brick_poses(color_img, maskModel)
    annotated_frame = registered_bricks.plot()  

     # visualize images if needed
    #cv2.imshow("Aria Image", color_img)
    #cv2.imshow("Gray Image", gray_img)
    #cv2.imshow("BBoxes", annotated_frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # extract timestamps and calculate gaze_points from pitch and yaw
    first_timestamp = intent_json.get('startTime_ns', [])
    last_timestamp = intent_json.get('endTime_ns', [])
    foc_boxes = []

    for ts in range(len(first_timestamp)):
        et_data = extract_yaw_pitch(et_csv_file_path, [first_timestamp[ts]], [last_timestamp[ts]] )
        filename = extract_timestamp(rgb_csv_file_path, [first_timestamp[0]])
        gaze_points = et_data[['gaze_point_x', 'gaze_point_y']].to_numpy()

        # get bounding boxes from YOLO
        if registered_bricks is None:
            print("No bricks detected, returning original image")
            return gray_img, None
        foc_box = find_most_focused_bbox(gaze_points, registered_bricks.boxes)
        print(f"Most focused bbox: {foc_box}")
        foc_boxes.append(foc_box)

    # Visualise the selected bricks
    color = color = (0, 255, 0) # brick to be manipulated -> green
    for foc_box in foc_boxes:
        x1, y1, x2, y2 = foc_box["coordinates"]
        cv2.rectangle(
            color_img,
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            color= color, 
            thickness=2        
        )
        color = (0, 0, 255) # brick to be placed on -> red
    cv2.imshow("Selected Bricks", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = [fx['coordinates'] for fx in foc_boxes]
    return gray_img, result, nlp_command


def extract_timestamp(csv_file, first_timestamp):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="#timestamp [ns]")
    start_row = df.iloc[(df['#timestamp [ns]'] - first_timestamp).abs().idxmin()]
    start_idx = start_row.name
    filename = df.loc[start_idx, 'filename']
    return filename


def extract_yaw_pitch(csv_file, first_timestamp, last_timestamp):
    # load CSV and sort by timestamps
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="tracking_timestamp_ns")
    
    # Find closest rows to first_timestamp and last_timestamp
    start_row = df.iloc[(df['tracking_timestamp_ns'] - first_timestamp).abs().idxmin()]
    end_row = df.iloc[(df['tracking_timestamp_ns'] - last_timestamp).abs().idxmin()]

    # get indices and slice and save data inbetween timestamps
    start_idx = start_row.name
    end_idx = end_row.name
    print(f"slice data between timestamps {start_idx} and {end_idx}")
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    result = df.loc[start_idx:end_idx, ['tracking_timestamp_ns', 'gaze_point_x', 'gaze_point_y']]
    return result


def get_brick_poses(color_image, maskModel):
    registered_bricks = maskModel(
        color_image, iou=0.9, verbose=False)[0]
    if not registered_bricks or registered_bricks[0] is None:
        print("Warnung: Keine Erkennung")
        return None
    return registered_bricks


def find_most_focused_bbox(gaze_points, result_boxes):
    focus_counter = Counter()
    bbox_lookup = {}
    distances_per_bbox = defaultdict(list)

    for gaze_point in gaze_points:
        focused = find_focused_bbox(gaze_point, result_boxes)
        if focused:
            key = focused["bbox_index"]
            focus_counter[key] += 1
            distances_per_bbox[key].append(focused["distance"])
            if key not in bbox_lookup:
                bbox_lookup[key] = focused

    if not focus_counter:
        return None

    most_common_key, count = focus_counter.most_common(1)[0]
    avg_distance = sum(distances_per_bbox[most_common_key]) / len(distances_per_bbox[most_common_key])
    result = bbox_lookup[most_common_key].copy()
    result["focus_count"] = count
    result["distance"] = avg_distance
    return result


def find_focused_bbox(gaze_point, result_boxes):
    x, y = gaze_point
    closest_bbox = None
    min_dist = float('inf')
    i=0
    threshold = 150
    
    # find focused bbox
    for brick in result_boxes:
        bbox_tensor = brick.xyxy[0]  # bbox coordinates as tensor
        x1, y1, x2, y2 = bbox_tensor.tolist()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance = math.hypot(center_x - x, center_y - y)
        i += 1
        if distance < min_dist and distance < threshold:
            min_dist = distance
            closest_bbox = {
                "confidence": brick.conf[0],
                "class": brick.cls[0],
                "bbox_index": i,
                "coordinates": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "distance": distance
            }
    return closest_bbox     


def publish_bbox(context, tool_call, bboxes, image_shape, ip, grab=False):

    # extract most important values from bbox into dictionary
    if grab:
        bbox_dicts = []
        for bbox in bboxes:
            bbox_tensor = bbox.xyxy[0]
            x1, y1, x2, y2 = bbox_tensor.tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height, width = image_shape[:2]
            flipped_center = (width - center_x, height - center_y)
            bbox_dict = {
                "confidence": bbox.conf[0],
                "class": bbox.cls[0],
                "coordinates": (x1, y1, x2, y2),
                "center": flipped_center,
            }
            bbox_dicts.append(bbox_dict)


    socket = context.socket(zmq.REQ)
    socket.connect(ip + ":5561")
    print("ZMQ socket connected to Panda3 PC")
    
    # distinguish between greb_brick, place_bricks (append coords) and others
    if grab:
        msg = {"function_name": [tool_call],
                "arguments": [bbx['center'] for bbx in bbox_dicts]
                }
        socket.send_json(msg)
        print("Bounding box of brick to grab sent to Panda3 PC...")
    else:
        socket.send_string(tool_call)

    # receive answer, whether tool call was succesfull

    success = socket.recv_json()
    if success:
        print("Function successfully executed by Panda")
    else:
        print("Error: Function not successfully executed by Panda")
    socket.close()


def match_features():
    srcpath = os.path.dirname(os.path.abspath(__file__))
    filepath = srcpath[:-4]  # Adjust path to project root
    data_path = os.path.join(filepath, "data")
    robot_pkg_ip = "tcp://127.0.0.1"
    context = zmq.Context()
    maskModel = YOLO(os.path.join(srcpath, "best.pt"))
    
    # 1. Initialize SuperGlue
    print("Initializing SuperGlue")
    matching = superglue_matching_init()

    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5560")
    print("ZMQ socket bound to port 5560")

    # receive inferred command from voice_controller
    print("Waiting for message from voice_controller script...\n")
    nlp_string = socket.recv_string()
    if nlp_string is None:
        print("No response received, return None")
        #return None
    else:
        print(f"Received String object as response: {nlp_string}")
    
    nlp_json = json.loads(nlp_string)

    socket.close()
    arg_i = 0
    for function_name in nlp_json["function_name"]:
        command_json = {"function_name": [function_name],
                        "arguments": []}
        if function_name == "grab_brick" or function_name == "place_brick" or function_name == "move_arm" or function_name == "turn_arm":
           command_json["arguments"] = nlp_json["arguments"][arg_i]
           arg_i +=1
        else:
            command_json["arguments"] = [None]

        # 2. get images and bounding boxes from Aria and Robo
        robo_img, bboxesrobo  = get_robo_img_bbox(context, maskModel, robot_pkg_ip) 
        print(f"Panda Intel: Image and coordinates of Bounding Boxes found: {bboxesrobo}")
        aria_bbox_response= get_aria_img_bbox(context, maskModel, command_json, data_path, robot_pkg_ip)  

        if aria_bbox_response is not None: 
            aria_img, bboxesaria, tool_call = aria_bbox_response

            # 3. Start matching features and compute bbox with gaze
            print("Start matching features...")
            aria_matched, robo_matched_list, mconf = superglue(matching, aria_img, robo_img, bboxesaria, data_path)
            print(f"matched Aria points: {aria_matched}")

            targeted_bboxes = []
            for robo_matched in robo_matched_list:
                matched_bbox, points_per_bbox = calculate_matching_points_in_box(robo_matched, bboxesrobo)
                targeted_bbox = bboxesrobo[matched_bbox]
                print(f"Matched bounding box index: {matched_bbox} with {points_per_bbox} points.")  
                targeted_bboxes.append(targeted_bbox)

            # 4. publish the Bbox to Panda3 PC for grasping
            publish_bbox(context, tool_call, targeted_bboxes, robo_img.shape, robot_pkg_ip, grab=True)