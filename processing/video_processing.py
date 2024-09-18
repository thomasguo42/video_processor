import torch
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import math
import os
from numpy.polynomial.polynomial import Polynomial


# Function to get the largest two human bounding boxes or user-selected ones
def get_largest_two_humans(results):
    largest_indices = []

    for r in results:
        boxes = []
        # Extract bounding box coordinates and calculate areas
        for i, box in enumerate(r.cpu().boxes.xyxy.numpy()):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            boxes.append((area, i, x1))

        # Sort boxes by area and get the largest two
        boxes = sorted(boxes, key=lambda x: x[0], reverse=True)[:2]

        # Handle cases with less than two people detected
        if len(boxes) < 2:
            # Append invalid indices for frames with fewer than two people
            largest_indices.append([-1, -1])
        else:
            # Sort the largest two boxes by their x1 coordinate (leftmost first)
            largest_two_boxes = sorted(boxes[:2], key=lambda x: x[2])
            largest_indices.append([largest_two_boxes[0][1], largest_two_boxes[1][1]])

    return largest_indices


def check(r, p, q):
    boxes = r.cpu().boxes.xyxy.numpy()
    ids = r.cpu().boxes.id.numpy()
    keys = r.cpu().keypoints.xy.numpy()
    keypoint_index = set()
    for idx, box_id in enumerate(ids):
        if box_id == p+1:
            x1e, y1e, x2e, y2e = boxes[idx]
        if box_id == q+1:
            x1f, y1f, x2f, y2f = boxes[idx]
    
    for d in range (len(r.cpu().keypoints.xy.numpy())):
        if (x1e < keys[d][14][0] < x2e) and (x1e < keys[d][13][0] < x2e):
            keypoint_index.add(d)
        elif (x1f < keys[d][14][0] < x2f) and (x1f < keys[d][13][0] < x2f):
            keypoint_index.add(d)
        elif len(keypoint_index)>2:
            break

    return keypoint_index

# Function to identify and correct abnormalities
def identify_and_correct_abnormalities(data, threshold=0.3):
    corrected_data = data.copy()
    n = len(data)
    i = 1

    while i < n - 1:
        if abs(data[i] - data[i - 1]) > threshold:
            start = i
            wrong = [data[i]]
            j = i
            k = 1

            while j < n - 1 and abs(data[j + 1] - data[j]) < 0.1:
                wrong.append(data[j + 1])
                j += 1
                k += 1
            
            end = j + 1
            if start > 0 and end < n:
                x0, y0 = start - 1, corrected_data[start - 1]
                x1, y1 = end, corrected_data[end]

                for t in range(start, end):
                    corrected_data[t] = y0 + (y1 - y0) * (t - x0) / (x1 - x0)
            
            i = end
        else:
            i += 1

    return corrected_data


def draw_boxes_on_first_frame(frame, results):
    # Convert the frame to RGB format for consistent coloring
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    for i, box in enumerate(results.cpu().boxes.xyxy.numpy()):
        x1, y1, x2, y2 = box
        
        # Draw a rectangle around the detected person
        cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=2)
        
        # Add the index of the bounding box
        cv2.putText(frame_rgb, str(i), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Return the modified image
    return frame_rgb

def get_human_indices_from_user(num_humans, first_frame_results, source):
    # Capture the first frame
    cap = cv2.VideoCapture(source)
    ret, first_frame = cap.read()
    cap.release()
    choose_frame = draw_boxes_on_first_frame(first_frame, first_frame_results)

    if num_humans > 2:
        print(f"More than 2 humans detected. Please input the indexes of the two humans to track (0 to {num_humans-1}):")
        

        return False, choose_frame
    else:
        return True, choose_frame

def process_video_and_extract_data(results, source, tracked_indices):
    left_xdata = {k: [] for k in range(7, 17)}
    left_ydata = {k: [] for k in range(7, 17)}
    right_xdata = {k: [] for k in range(7, 17)}
    right_ydata = {k: [] for k in range(7, 17)}
    checker_list = []
    video_angle = ''

    p, q = tracked_indices
    print ('getting value')
    for i in range (len(results)):
        try:
            values = [results[i].cpu().keypoints.xy.numpy()[p][16][0], results[i].cpu().keypoints.xy.numpy()[p][15][0], 
                      results[i].cpu().keypoints.xy.numpy()[q][16][0], results[i].cpu().keypoints.xy.numpy()[q][15][0]]
            print ('get value')
            sorted_values = sorted(values, reverse=True)
            b = sorted_values[1]
            a = sorted_values[2]
            c = abs((b-a)/4)
            break
        except:
            continue

    for i in range (len(results)):
        try:
            current_boxes = results[i].cpu().boxes.xyxy.numpy()
            left_box_area = (current_boxes[p][2] - current_boxes[p][0]) * (current_boxes[p][3] - current_boxes[p][1])
            right_box_area = (current_boxes[q][2] - current_boxes[q][0]) * (current_boxes[q][3] - current_boxes[q][1])
            break
        except:
            continue

    if left_box_area >= 1.75 * right_box_area:
        video_angle = 'left'
    elif right_box_area >= 1.75 * left_box_area:
        video_angle = 'right'
    else:
        video_angle = 'middle'

    i = 0
    for r in tqdm(results):
        try:
            keys = r.cpu().keypoints.xy.numpy()
            current_boxes = r.cpu().boxes.xyxy.numpy()
            tracked_boxes = [current_boxes[idx] for idx in tracked_indices]

            keypoint_index = check(r, p, q)
            keypoint_index_list = list(keypoint_index)

            if len(keypoint_index_list) == 2:
                e = keypoint_index_list[0]
                f = keypoint_index_list[1]

            for j in range(7, 17, 2):
                if keys[e][j][0] < keys[f][j][0]:
                    if keys[e][11][0] < keys[e][12][0]:
                        left_xdata[j].append(keys[e][j][0] / c)
                        left_ydata[j].append(keys[e][j][1] / c)
                        left_xdata[j + 1].append(keys[e][j + 1][0] / c)
                        left_ydata[j + 1].append(keys[e][j + 1][1] / c)
                    else:
                        left_xdata[j].append(keys[e][j + 1][0] / c)
                        left_ydata[j].append(keys[e][j + 1][1] / c)
                        left_xdata[j + 1].append(keys[e][j][0] / c)
                        left_ydata[j + 1].append(keys[e][j][1] / c)
    
                    if keys[f][11][0] < keys[f][12][0]:
                        right_xdata[j].append(keys[f][j + 1][0] / c)
                        right_ydata[j].append(keys[f][j + 1][1] / c)
                        right_xdata[j + 1].append(keys[f][j][0] / c)
                        right_ydata[j + 1].append(keys[f][j][1] / c)
                    else:
                        right_xdata[j].append(keys[f][j][0] / c)
                        right_ydata[j].append(keys[f][j][1] / c)
                        right_xdata[j + 1].append(keys[f][j + 1][0] / c)
                        right_ydata[j + 1].append(keys[f][j + 1][1] / c)
                else:
                    if keys[f][11][0] < keys[f][12][0]:
                        left_xdata[j].append(keys[f][j][0] / c)
                        left_ydata[j].append(keys[f][j][1] / c)
                        left_xdata[j + 1].append(keys[f][j + 1][0] / c)
                        left_ydata[j + 1].append(keys[f][j + 1][1] / c)
                    else:
                        left_xdata[j].append(keys[f][j + 1][0] / c)
                        left_ydata[j].append(keys[f][j + 1][1] / c)
                        left_xdata[j + 1].append(keys[f][j][0] / c)
                        left_ydata[j + 1].append(keys[f][j][1] / c)
    
                    if keys[e][11][0] < keys[e][12][0]:
                        right_xdata[j].append(keys[e][j + 1][0] / c)
                        right_ydata[j].append(keys[e][j + 1][1] / c)
                        right_xdata[j + 1].append(keys[e][j][0] / c)
                        right_ydata[j + 1].append(keys[e][j][1] / c)
                    else:
                        right_xdata[j].append(keys[e][j][0] / c)
                        right_ydata[j].append(keys[e][j][1] / c)
                        right_xdata[j + 1].append(keys[e][j + 1][0] / c)
                        right_ydata[j + 1].append(keys[e][j + 1][1] / c)
            i+=1

        except:
            i+=1
            continue

    return left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle



import cv2
import os

import os
from django.conf import settings

import os
import cv2
from moviepy.editor import VideoFileClip

def generate_video_with_keypoints(results, source_video_path, tracked_indices, left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, output_prefix, video_id):
    # Load the video using moviepy
    video = VideoFileClip(source_video_path)
    fps = max(video.fps // 2, 15)  # Reduce FPS by half, but ensure it's at least 15
    
    # Get the video resolution and reduce it for smaller file size
    width, height = video.size
    width = width // 2  # Reduce width by half
    height = height // 2  # Reduce height by half
    
    # Create output directory for saving the video
    output_dir = os.path.join(settings.MEDIA_ROOT, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output video path
    video_output_filename = f'output_with_keypoints_{video_id}.mp4'
    video_output_path = os.path.join(output_dir, video_output_filename)
    
    # OpenCV VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))  # Use video size from moviepy
    
    # Capture video frames using OpenCV
    cap = cv2.VideoCapture(source_video_path)
    frameNr = 0
    
    # Process each frame and overlay keypoints
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the corresponding row for this frame from the dataframes
        left_x_frame_row = left_xdata_df[left_xdata_df['Frame'] == frameNr]
        right_x_frame_row = right_xdata_df[right_xdata_df['Frame'] == frameNr]
        left_y_frame_row = left_ydata_df[left_xdata_df['Frame'] == frameNr]
        right_y_frame_row = right_ydata_df[right_xdata_df['Frame'] == frameNr]

        # If there is no matching row for this frame, skip to the next frame
        if left_x_frame_row.empty or right_x_frame_row.empty:
            frameNr += 1
            continue
        
        # Add key points for the two chosen fencers
        for j in range(7, 17):  # You are using columns 7 to 16 for keypoints
            try:
                # Plot keypoints for the left fencer (match the 'Frame' row)
                left_x = int(left_x_frame_row[j].values[0] * c)
                left_y = int(left_y_frame_row[j].values[0] * c)
                cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'{j}', (left_x + 10, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Plot keypoints for the right fencer (match the 'Frame' row)
                right_x = int(right_x_frame_row[j].values[0] * c)
                right_y = int(right_y_frame_row[j].values[0] * c)
                cv2.circle(frame, (right_x, right_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'{j}', (right_x + 10, right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            except KeyError:
                # In case any keypoint data is missing
                continue
        
        # Write the modified frame to the output video
        out.write(frame)
        frameNr += 1
    
    cap.release()
    out.release()
    
    # Create the media URL for the output video
    video_output_url = os.path.join(settings.MEDIA_URL, output_prefix, video_output_filename)
    
    return video_output_url  # Return the URL for the saved video



# Function to update zero_indices while skipping specified columns
def update_zero_indices(df, zero_indices, columns_to_skip):
    # Filter out the columns that need to be skipped
    columns_to_check = [col for col in df.columns if col not in columns_to_skip]
    
    # Iterate over the rows and check for zeros in the specified columns
    for idx in df.index:
        if (df.loc[idx, columns_to_check] == 0).any():
            zero_indices.add(idx)

def process_video(source):
    if isinstance(source, list):  # If the source is a list, it's a single frame
        # Process a single frame
        frame = source[0]
        model = YOLO("yolov8m-pose.pt")
        results = model([frame])  # Process the single frame
        first_frame_results = results[0]
        choose_frame = draw_boxes_on_first_frame(frame, first_frame_results)
        num_humans = len(first_frame_results.cpu().boxes.xyxy.numpy())
        if_proceed = num_humans == 2
        return results, first_frame_results, num_humans, if_proceed, choose_frame
    else:
        # Process the entire video
        model = YOLO("yolov8m-pose.pt")
        results = model.track(source, persist=True)
        first_frame_results = results[0]
        num_humans = len(first_frame_results.cpu().boxes.xyxy.numpy())
        if_proceed = num_humans == 2
        choose_frame = draw_boxes_on_first_frame(results[0].orig_img, first_frame_results)
        return results, first_frame_results, num_humans, if_proceed, choose_frame


def extract_data(source, results, first_frame_results, tracked_indices):
    left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle = process_video_and_extract_data(results, source, tracked_indices)
    print ('Update zeros')
    left_xdata_df = pd.DataFrame(left_xdata)
    left_ydata_df = pd.DataFrame(left_ydata)
    right_xdata_df = pd.DataFrame(right_xdata)
    right_ydata_df = pd.DataFrame(right_ydata)
    checker_list_df = pd.DataFrame(checker_list)
    video_angle_df = pd.DataFrame([video_angle], columns=['Video Angle'])
    
    # Add original frame index as a column
    left_xdata_df['Frame'] = left_xdata_df.index
    left_ydata_df['Frame'] = left_ydata_df.index
    right_xdata_df['Frame'] = right_xdata_df.index
    right_ydata_df['Frame'] = right_ydata_df.index
    
    # Initialize a set to collect indices with zero values
    zero_indices = set()
    
    # Define the columns to skip for zero checks
    columns_to_skip = {7, 8, 9, 10, 11, 12, 13, 14}
    
    # Update zero_indices for each DataFrame
    update_zero_indices(left_xdata_df.drop(columns=['Frame']), zero_indices, columns_to_skip)
    update_zero_indices(left_ydata_df.drop(columns=['Frame']), zero_indices, columns_to_skip)
    update_zero_indices(right_xdata_df.drop(columns=['Frame']), zero_indices, columns_to_skip)
    update_zero_indices(right_ydata_df.drop(columns=['Frame']), zero_indices, columns_to_skip)
    
    
    # Remove the rows with zero values from all DataFrames, excluding rows to skip
    left_xdata_df = left_xdata_df.drop(index=zero_indices)
    left_ydata_df = left_ydata_df.drop(index=zero_indices)
    right_xdata_df = right_xdata_df.drop(index=zero_indices)
    right_ydata_df = right_ydata_df.drop(index=zero_indices)

    return left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, checker_list_df, video_angle_df

def extract_keypoints(xdata, ydata, index):
    return {
        'hipback': (xdata.loc[index, 11], ydata.loc[index, 11]),
        'kneeback': (xdata.loc[index, 13], ydata.loc[index, 13]),
        'ankleback': (xdata.loc[index, 15], ydata.loc[index, 15]),
        'hipfront': (xdata.loc[index, 12], ydata.loc[index, 12]),
        'kneefront': (xdata.loc[index, 14], ydata.loc[index, 14]),
        'anklefront': (xdata.loc[index, 16], ydata.loc[index, 16]),
        'handfront': (xdata.loc[index, 10], ydata.loc[index, 10])
    }

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ab = a - b
    cb = c - b

    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to calculate all relevant angles for a fencer
def calculate_fencer_angles(fencer_keypoints):
    return {
        'back_knee_angle': calculate_angle(fencer_keypoints['ankleback'], fencer_keypoints['kneeback'], fencer_keypoints['hipback']),
        'back_hip_angle': calculate_angle(fencer_keypoints['kneeback'], fencer_keypoints['hipback'], fencer_keypoints['hipfront']),
        'front_hip_angle': calculate_angle(fencer_keypoints['hipback'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
        'front_knee_angle': calculate_angle(fencer_keypoints['hipfront'], fencer_keypoints['kneefront'], fencer_keypoints['anklefront']),
        'hand_hip_angle': calculate_angle(fencer_keypoints['handfront'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
    }

def load_start_compare_data(start_file_path=None):
    if start_file_path is None:
        start_file_path = os.path.join('processing', 'compare_data', 'start compare.xlsx')
  
    left_xdata_df_start = pd.read_excel(start_file_path, sheet_name='left_xdata')
    left_ydata_df_start = pd.read_excel(start_file_path, sheet_name='left_ydata')
    right_xdata_df_start = pd.read_excel(start_file_path, sheet_name='right_xdata')
    right_ydata_df_start = pd.read_excel(start_file_path, sheet_name='right_ydata')

    left_fencer_keypoints_start = [
        extract_keypoints(left_xdata_df_start, left_ydata_df_start, i) for i in range(3)
    ]
    right_fencer_keypoints_start = [
        extract_keypoints(right_xdata_df_start, right_ydata_df_start, i) for i in range(3)
    ]

    left_angles_start = [calculate_fencer_angles(kp) for kp in left_fencer_keypoints_start]
    right_angles_start = [calculate_fencer_angles(kp) for kp in right_fencer_keypoints_start]

    return left_angles_start, right_angles_start

def calculate_angle_differences(left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, video_angle_df):
    left_angles_start, right_angles_start = load_start_compare_data()

    angle_differences = {
        'Frame': [],
        'back_knee_angle_left_diff': [],
        'back_hip_angle_left_diff': [],
        'front_hip_angle_left_diff': [],
        'front_knee_angle_left_diff': [],
        'hand_hip_angle_left_diff': [],
        'back_knee_angle_right_diff': [],
        'back_hip_angle_right_diff': [],
        'front_hip_angle_right_diff': [],
        'front_knee_angle_right_diff': [],
        'hand_hip_angle_right_diff': []
    }

    for index in range(len(left_xdata_new_df)):
        try:
            left_fencer_keypoints = extract_keypoints(left_xdata_new_df, left_ydata_new_df, index)
            right_fencer_keypoints = extract_keypoints(right_xdata_new_df, right_ydata_new_df, index)
            
            left_angles = calculate_fencer_angles(left_fencer_keypoints)
            right_angles = calculate_fencer_angles(right_fencer_keypoints)
    
            angle = video_angle_df.loc[0, 'Video Angle']
            if angle == 'middle':
                left_start_angles = left_angles_start[0]
                right_start_angles = right_angles_start[0]
            elif angle == 'left':
                left_start_angles = left_angles_start[1]
                right_start_angles = right_angles_start[1]
            else:
                left_start_angles = left_angles_start[2]
                right_start_angles = right_angles_start[2]
            
            for angle_name in left_start_angles.keys():
                left_diff = abs(left_angles[angle_name] - left_start_angles[angle_name])
                right_diff = abs(right_angles[angle_name] - right_start_angles[angle_name])
                angle_differences[angle_name + '_left_diff'].append(left_diff)
                angle_differences[angle_name + '_right_diff'].append(right_diff)
            
            angle_differences['Frame'].append(left_xdata_new_df.loc[index, 'Frame'])
        except:
            continue

    angle_differences_df = pd.DataFrame(angle_differences)
    angle_differences_df['Total_Diff'] = angle_differences_df.drop(columns=['Frame']).sum(axis=1)

    # Identify the threshold for the lowest 20% of the summed differences
    threshold = angle_differences_df['Total_Diff'].quantile(0.20)
    
    # Find segments where all values are below the threshold
    segments = []
    current_segment = []
    
    for i, row in angle_differences_df.iterrows():
        if row['Total_Diff'] <= threshold:
            current_segment.append(row['Frame'])
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    
    # Add the last segment if it meets the criteria
    if current_segment:
        segments.append(current_segment)

    return segments

def calculate_jerk(acceleration):
    return np.diff(acceleration)

def calculate_jerk_based_frames(left_xdata_new_df, right_xdata_new_df):
    # Calculate velocities (rate of change of x-coordinates) for keypoint 16
    left_velocity = np.diff(left_xdata_new_df[16].values)
    right_velocity = -np.diff(right_xdata_new_df[16].values)  # Invert velocity for the right fencer
    
    # Calculate accelerations (rate of change of velocities)
    left_acceleration = np.diff(left_velocity)
    right_acceleration = np.diff(right_velocity)
    
    # Calculate the difference in acceleration (jerk)
    left_jerk = calculate_jerk(left_acceleration)
    right_jerk = calculate_jerk(right_acceleration)
    
    # Get the indices of the three largest decreases in acceleration (jerk) for left and right fencers
    min_left_jerk_indices = np.argsort(left_jerk)[:3]
    min_right_jerk_indices = np.argsort(right_jerk)[:3]
    
    min_left_acceleration_indices = np.argsort(left_acceleration)[:3]
    min_right_acceleration_indices = np.argsort(right_acceleration)[:3]
    
    # Get the corresponding frame numbers for these indices
    min_left_jerk_frames = left_xdata_new_df.iloc[min_left_jerk_indices]['Frame'].values
    min_right_jerk_frames = right_xdata_new_df.iloc[min_right_jerk_indices]['Frame'].values
    
    min_left_jerk_frames = left_xdata_new_df.iloc[min_left_acceleration_indices]['Frame'].values
    min_right_jerk_frames = right_xdata_new_df.iloc[min_right_acceleration_indices]['Frame'].values
    
    # Combine these frames to have 6 frames
    all_jerk_frames = np.concatenate((min_left_jerk_frames, min_right_jerk_frames))

    return all_jerk_frames

def merge_consecutive_numbers(numbers):
    if not numbers:
        return []

    # Sort the list first to ensure consecutive order
    numbers = sorted(numbers)
    
    merged_segments = []
    start = numbers[0]
    end = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            merged_segments.append((start, end))
            start = numbers[i]
            end = numbers[i]

    # Append the last segment
    merged_segments.append((start, end))
    
    return merged_segments

def find_longest_segment(segments):
    # Calculate the length of each segment
    segment_lengths = [(seg, seg[1] - seg[0] + 1) for seg in segments]
    
    # Identify the segment with the maximum length
    if segment_lengths:
        longest_segment = max(segment_lengths, key=lambda x: x[1])[0]
        end_frame = longest_segment[0] + math.ceil((longest_segment[1] - longest_segment[0]) / 2)
        return end_frame
    else:
        return None

def filter_segments_by_jerk_frames(segments, jerk_frames):
    filtered_segments = []
    for seg in segments:
        if any(frame in jerk_frames for frame in range(seg[0] - 5, seg[1] + 5)):
            filtered_segments.append(seg)
    return filtered_segments

def find_end_frame(left_xdata_new_df, right_xdata_new_df, all_jerk_frames):
    # Assume 'a' is already defined as:
    a = abs(right_xdata_new_df[16] - left_xdata_new_df[16])
    
    end_frames = []
    for i in range(len(a)):
        try:
            if 0 < a[i] < 0.5:  
                end_frames.append(i)
        except:
            continue
    if not end_frames:
        thresh = a.quantile(0.10)
        for i in range(len(a)):
            try:
                if 0 < a[i] < thresh:  
                    end_frames.append(i)
            except:
                continue
    end_segments = merge_consecutive_numbers(end_frames)
    
    
    # Filter end_segments based on all_jerk_frames
    filtered_end_segments = filter_segments_by_jerk_frames(end_segments, all_jerk_frames)
    print (end_segments, all_jerk_frames, filtered_end_segments)
    
    if not filtered_end_segments:
        filtered_end_segments = end_segments[-1]
    
    # Check if there is only one segment
    if isinstance(filtered_end_segments, tuple):
        # If it's a tuple (a single segment), we can directly use it
        longest_segment = filtered_end_segments
        end_frame = longest_segment[0] + math.ceil((longest_segment[1] - longest_segment[0]) / 2)
    else:
        # Otherwise, find the longest segment from the list of segments
        end_frame = find_longest_segment(filtered_end_segments)

    end_frame = end_frame - 7

    return end_frame

def calculate_center_of_mass(left_xdata_df_normalized, right_xdata_df_normalized):
    left_x_coordinates = left_xdata_df_normalized.iloc[:, :-1].copy()
    num_columns = left_x_coordinates.shape[1] - 1
    remaining_weight = 0.2
    even_weight = remaining_weight / num_columns

    weights = pd.Series(even_weight, index=left_x_coordinates.columns)
    weights[16] = 0.8

    left_xdata_df_normalized['Center_of_Mass_X'] = (left_x_coordinates * weights).sum(axis=1)

    right_x_coordinates = right_xdata_df_normalized.iloc[:, :-1].copy()
    num_columns = right_x_coordinates.shape[1] - 1
    even_weight = remaining_weight / num_columns

    weights = pd.Series(even_weight, index=right_x_coordinates.columns)
    weights[16] = 0.8

    right_xdata_df_normalized['Center_of_Mass_X'] = (right_x_coordinates * weights).sum(axis=1)

    return left_xdata_df_normalized, right_xdata_df_normalized

def merge_consecutive_numbers(numbers):
    if not numbers:
        return []

    # Sort the list first to ensure consecutive order
    numbers = sorted(numbers)
    
    merged_segments = []
    start = numbers[0]
    end = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            merged_segments.append((start, end))
            start = numbers[i]
            end = numbers[i]

    # Append the last segment
    merged_segments.append((start, end))
    
    return merged_segments

def process_segments(segments):
    # Step 1: Delete all segments that start within the first 5 frames
    segments = [segment for segment in segments if segment[0] > 5]

    # Step 2: Delete all segments with a length of less than 8
    segments = [segment for segment in segments if segment[1] - segment[0] + 1 >= 8]

    # Step 3: Sort the remaining segments by their end frame
    segments.sort(key=lambda x: x[1])

    return segments

def merge_close_segments(segments, max_gap=2):
    if not segments:
        return []  # Return an empty list if there are no segments

    # Sort segments by their starting point
    segments.sort(key=lambda seg: seg[0])

    merged_segments = []
    current_segment = segments[0]

    for seg in segments[1:]:
        if seg[0] - current_segment[1] <= max_gap:
            if (seg[1]-seg[0]) >= 2 or (current_segment[1]-current_segment[0]) >= 2:
                # Merge the segments by extending the end of the current segment
                current_segment = (current_segment[0], max(current_segment[1], seg[1]))
        else:
            # If segments are not close enough, save the current segment and start a new one
            merged_segments.append(current_segment)
            current_segment = seg

    # Append the last segment
    merged_segments.append(current_segment)
    
    return merged_segments

def find_overlapping_leftright_segments(segments1, segments2, min_overlap=7):
    overlapping_segments = []
    
    for seg1 in segments1:
        for seg2 in segments2:
            
            start_overlap = max(seg1[0], seg2[0])
            end_overlap = min(seg1[1], seg2[1])

            # Calculate the number of overlapping frames
            overlap = end_overlap - start_overlap + 1
            
            if overlap >= min_overlap:
                overlapping_segments.append((start_overlap, end_overlap))
                
    
    return overlapping_segments

def find_overlapping_segments(segments1, segments2, min_overlap=7):
    overlapping_segments = []
    involved_segments = []
    
    for seg1 in segments1:
        for seg2 in segments2:
            start_overlap = max(seg1[0], seg2[0])
            end_overlap = min(seg1[1], seg2[-1])

            # Calculate the number of overlapping frames
            overlap = end_overlap - start_overlap + 1
            
            if overlap >= min_overlap:
                overlapping_segments.append((start_overlap, end_overlap))
                involved_segments.append(seg1)
                
    
    return overlapping_segments, involved_segments

def find_latest_end(left_xdata_df_normalized, right_xdata_df_normalized, video_angle_df, segments, end_frame):
    left_xdata_df_normalized, right_xdata_df_normalized = calculate_center_of_mass(left_xdata_df_normalized, right_xdata_df_normalized)
    a = np.diff(-right_xdata_df_normalized['Center_of_Mass_X'])
    c = pd.DataFrame(a)
    
    b = np.diff(left_xdata_df_normalized['Center_of_Mass_X'])
    d = pd.DataFrame(b)
    
    left = []
    right = []

    video_angle = video_angle_df.loc[0, 'Video Angle']
    
    if video_angle == 'middle':
        right_threshold = 0.06
        left_threshold = 0.06
    elif video_angle == 'right':
        right_threshold = 0.1
        left_threshold = 0.02
    elif video_angle == 'left':
        right_threshold = 0.02
        left_threshold = 0.1
    
    for i in range (len(c[0])):
        if c[0][i] < right_threshold:
            right.append(i)
            
    for i in range (len(d[0])):
        if d[0][i] < left_threshold:
            left.append(i)
    
    left_segments = merge_consecutive_numbers(left)
    right_segments = merge_consecutive_numbers(right)
    
    left_merged_segments = merge_close_segments(left_segments)
    right_merged_segments = merge_close_segments(right_segments)

    merged_segments = find_overlapping_leftright_segments(left_merged_segments, right_merged_segments)

    # Filter out segments with length less than 7
    filtered_segments = [segment for segment in segments if len(segment) >= 7]
    
    

    # Use the function to find overlapping segments
    overlapping_segments, involved_segments = find_overlapping_segments(merged_segments, filtered_segments)
    
    if involved_segments:
        latest_end = max(seg[1] for seg in involved_segments)
    else:
        latest_end = 0

    if (end_frame - latest_end)<10:
        latest_end = 0

    return latest_end

def cut_video(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, video_angle_df):
    segments = calculate_angle_differences(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, video_angle_df)
    all_jerk_frames = calculate_jerk_based_frames(left_xdata_df, right_xdata_df)
    end_frame = find_end_frame(left_xdata_df, right_xdata_df, all_jerk_frames)
    latest_end = find_latest_end(left_xdata_df, right_xdata_df, video_angle_df, segments, end_frame)

    print(f"Latest End: {latest_end}, End Frame: {end_frame}")
    
    left_xdata_new_df = left_xdata_df[(left_xdata_df['Frame'] >= latest_end) & (left_xdata_df['Frame'] <= end_frame)]
    left_ydata_new_df = left_ydata_df[(left_ydata_df['Frame'] >= latest_end) & (left_ydata_df['Frame'] <= end_frame)]
    right_xdata_new_df = right_xdata_df[(right_xdata_df['Frame'] >= latest_end) & (right_xdata_df['Frame'] <= end_frame)]
    right_ydata_new_df = right_ydata_df[(right_ydata_df['Frame'] >= latest_end) & (right_ydata_df['Frame'] <= end_frame)]

    print(f"Left X DataFrame Indices After: {left_xdata_new_df['Frame'].tolist()}")

    return left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, latest_end, end_frame

import pickle
def load_all_data_from_single_pickle(input_path):
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def find_pause(left_xdata_df_normalized, right_xdata_df_normalized, video_angle_df):
    left_xdata_df_normalized, right_xdata_df_normalized = calculate_center_of_mass(left_xdata_df_normalized, right_xdata_df_normalized)
    a = np.diff(-right_xdata_df_normalized['Center_of_Mass_X'])
    c = pd.DataFrame(a)
    
    b = np.diff(left_xdata_df_normalized['Center_of_Mass_X'])
    d = pd.DataFrame(b)
    
    left = []
    right = []

    video_angle = video_angle_df.loc[0, 'Video Angle']
    
    if video_angle == 'middle':
        right_threshold = 0.08
        left_threshold = 0.08
    elif video_angle == 'right':
        right_threshold = 0.12
        left_threshold = 0.04
    elif video_angle == 'left':
        right_threshold = 0.04
        left_threshold = 0.12
    
    for i in range (len(c[0])):
        if c[0][i] < right_threshold:
            right.append(i)
            
    for i in range (len(d[0])):
        if d[0][i] < left_threshold:
            left.append(i)
    
    left_segments = merge_consecutive_numbers(left)
    right_segments = merge_consecutive_numbers(right)
    
    left_merged_segments = merge_close_segments(left_segments)
    right_merged_segments = merge_close_segments(right_segments)

    left_processed_segments = process_segments(left_merged_segments)
    right_processed_segments = process_segments(right_merged_segments)
    
    left_last = 0
    left_last_first = 0
    right_last = 0
    right_last_first = 0
    
    if left_processed_segments:
        left_processed_segments.sort(key=lambda seg: seg[0])
        left_last = left_processed_segments[-1][1]
        left_last_first = left_processed_segments[-1][0]
    
    if right_processed_segments:
        right_processed_segments.sort(key=lambda seg: seg[0])
        right_last = right_processed_segments[-1][1]
        right_last_first = right_processed_segments[-1][0]

    return left_last, left_last_first, right_last, right_last_first, left_processed_segments, right_processed_segments

def calculate_velocity_acceleration(data):
    # Fit a second-degree polynomial to the data
    x = np.arange(len(data))
    poly = Polynomial.fit(x, data, 2)

    # Calculate velocity as the first derivative of the polynomial
    velocity = poly.deriv(1)(x)

    # Calculate acceleration as the second derivative of the polynomial
    acceleration = poly.deriv(2)(x)

    return velocity, acceleration

def calculate_composite_scores(xdata, keypoints=[8, 10, 16], velocity_weight=0.8, acceleration_weight=0.2):
    composite_scores = []

    for k in keypoints:
        velocity, acceleration = calculate_velocity_acceleration(xdata[k])

        # Calculate the composite score for each frame
        composite_score = (velocity_weight * np.abs(velocity)) + (acceleration_weight * np.abs(acceleration))
        composite_scores.append(composite_score)

    # Average the scores across keypoints if there are multiple keypoints
    composite_scores = np.mean(composite_scores, axis=0)

    return composite_scores

def determine_frame_winner(left_scores, right_scores):
    # Count how many frames each fencer wins
    left_wins = np.sum(left_scores > right_scores)
    right_wins = np.sum(right_scores > left_scores)

    # Determine the winner based on the number of frames won
    if left_wins > right_wins:
        return 'left', left_wins, right_wins
    elif right_wins > left_wins:
        return 'right', left_wins, right_wins
    else:
        return 'draw', left_wins, right_wins

def find_rule_winner(left_xdata_new_df, right_xdata_new_df, left_last, left_last_first, right_last, right_last_first, left_processed_segments, right_processed_segments):
    left_composite_scores = calculate_composite_scores(left_xdata_new_df)
    right_composite_scores = calculate_composite_scores(-right_xdata_new_df)  # Invert direction for the right fencer

    mean_left_composite_scores = np.mean(left_composite_scores)
    mean_right_composite_scores = np.mean(right_composite_scores)

    if mean_left_composite_scores > mean_right_composite_scores:
        speed_winner = 'left'
    else:
        speed_winner = 'right'
    # Determine the frame-by-frame winner
    frame_winner, left_wins, right_wins = determine_frame_winner(left_composite_scores, right_composite_scores)
    
    if (not left_processed_segments) and (not right_processed_segments):
        rule_winner = speed_winner
    else:
        if left_last > right_last:
            rule_winner = 'right'
        elif left_last < right_last:
            rule_winner = 'left'
        elif left_last == right_last:
            if left_last_first > right_last_first:
                rule_winner = 'right'
            else:
                rule_winner = 'left'

    return rule_winner

def normalize_by_first_value(df):
    normalized_df = df.copy()
    for column in df.columns:
        first_value = df[column].iloc[0]
        normalized_df[column] = df[column] - first_value        
    return normalized_df


def calculate_difference(df1, df2, columns_to_compare):
    min_rows = min(len(df1), len(df2))
    # Use slicing directly and avoid unnecessary copying
    df1_trimmed = df1.iloc[-min_rows:, columns_to_compare]
    df2_trimmed = df2.iloc[-min_rows:, columns_to_compare]
    # Convert to numpy arrays for fast element-wise operations
    difference = np.abs(df1_trimmed.to_numpy() - df2_trimmed.to_numpy()).sum() / min_rows
 
    return difference


def compare_to_normalized_data(new_left_xdata_df, new_left_ydata_df, new_right_xdata_df, new_right_ydata_df, normalized_files_data, columns_to_compare):
    # Normalize the new DataFrames
    new_left_xdata_df = normalize_by_first_value(new_left_xdata_df)
    new_left_ydata_df = normalize_by_first_value(new_left_ydata_df)
    new_right_xdata_df = normalize_by_first_value(new_right_xdata_df)
    new_right_ydata_df = normalize_by_first_value(new_right_ydata_df)

    results = []
    for relative_path, (left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df) in tqdm(normalized_files_data.items(), desc="Comparing files"):
        try:
            difference = 0
            difference += calculate_difference(new_left_xdata_df, left_xdata_df, columns_to_compare)
            difference += calculate_difference(new_left_ydata_df, left_ydata_df, columns_to_compare)
            difference += calculate_difference(new_right_xdata_df, right_xdata_df, columns_to_compare)
            difference += calculate_difference(new_right_ydata_df, right_ydata_df, columns_to_compare)
            
            results.append((relative_path, difference))
        except Exception as e:
            print(f"Error comparing {relative_path}: {e}")
            continue
    
    results.sort(key=lambda x: x[1])
    
    return results[:11]


def find_ai_winner(new_left_xdata_df, new_left_ydata_df, new_right_xdata_df, new_right_ydata_df, normalized_files_data):
    columns_to_compare = [3, 6, 7, 8, 9]
    files = compare_to_normalized_data(new_left_xdata_df, new_left_ydata_df, new_right_xdata_df, new_right_ydata_df, normalized_files_data, columns_to_compare)
    left_count = sum('left' in os.path.basename(file) for file, _ in files)
    right_count = sum('right' in os.path.basename(file) for file, _ in files)

    total_files = len(files)
    left_percentage = (left_count / total_files) * 100
    right_percentage = (right_count / total_files) * 100

    if left_count > right_count:
        winner_ai = 'left'
    elif right_count > left_count:
        winner_ai = 'right'
    else:
        winner_ai = 'sim'

    print(f"'Left' files: {left_percentage:.2f}%")
    print(f"'Right' files: {right_percentage:.2f}%")

    return winner_ai, left_percentage, right_percentage

from django.conf import settings
import os

def find_winner(left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, video_angle_df):
    print ('loading pickle')
    output_pickle_file = os.path.join(settings.BASE_DIR, 'processing', 'mega_data_all.pkl')
    print ('finish loading')
    normalized_files_data = load_all_data_from_single_pickle(output_pickle_file)

    left_last, left_last_first, right_last, right_last_first, left_processed_segments, right_processed_segments = find_pause(left_xdata_new_df, right_xdata_new_df, video_angle_df)
    rule_winner = find_rule_winner(left_xdata_new_df, right_xdata_new_df, left_last, left_last_first, right_last, right_last_first, left_processed_segments, right_processed_segments)
    print ('running AI')
    winner_ai, left_percentage, right_percentage = find_ai_winner(left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, normalized_files_data)
    print ('finish running')
    segment_length = []
    for i in range (len(left_processed_segments)):
        segment_length.append(left_processed_segments[i][1]-left_processed_segments[i][0])
    
    for i in range (len(right_processed_segments)):
        segment_length.append(right_processed_segments[i][1]-right_processed_segments[i][0])
    
    if len(segment_length) != 0:
        max_length = max(segment_length)
    else:
        max_length = 0
                         
    
    if (len(left_xdata_new_df[16])) > 100:
        print (1)
        final_winner = winner_ai
    elif (len(left_xdata_new_df[16])) <= 100 and max_length>20:
        print (2)
        final_winner = winner_ai
    else:
        print (3)
        final_winner = rule_winner

    return final_winner, left_percentage, right_percentage, left_processed_segments, right_processed_segments

import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp


def get_frame_intervals(segments, frame_data):
    frame_intervals = []
    for seg in segments:
        start_frame = frame_data.iloc[seg[0]]
        end_frame = frame_data.iloc[seg[1]]
        frame_intervals.append((start_frame, end_frame))
    return frame_intervals


def cut_video_segments(source_video_path, frame_intervals, output_prefix, video_id):
    # Load the video
    video = mp.VideoFileClip(source_video_path)
    fps = video.fps

    # Create a directory for the output segments inside MEDIA_ROOT
    output_dir = os.path.join(settings.MEDIA_ROOT, output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    segment_urls = []

    # Process each interval
    for i, (start_frame, end_frame) in enumerate(frame_intervals):
        start_time = start_frame / fps
        end_time = end_frame / fps
        segment = video.subclip(start_time, end_time)

        # Save the segment to the output directory
        segment_output_filename = f'segment_{i+1}_{video_id}.mp4'
        segment_output_path = os.path.join(output_dir, segment_output_filename)
        segment.write_videofile(segment_output_path, codec="libx264")

        # Create the media URL for this segment and append it to the list
        segment_url = os.path.join(settings.MEDIA_URL, output_prefix, segment_output_filename)
        segment_urls.append(segment_url)

    video.close()
    return segment_urls


def calculate_velocity_acceleration(data):
    # Fit a second-degree polynomial to the data
    x = np.arange(len(data))
    poly = Polynomial.fit(x, data, 2)

    # Calculate velocity as the first derivative of the polynomial
    velocity = poly.deriv(1)(x)

    # Calculate acceleration as the second derivative of the polynomial
    acceleration = poly.deriv(2)(x)

    return velocity, acceleration
