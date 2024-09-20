from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from .forms import VideoUploadForm
from .models import VideoUpload
import os
from .video_processing import process_video, extract_data, get_largest_two_humans, cut_video
import matplotlib.pyplot as plt
import cv2
from django.http import JsonResponse


#上传视频
def video_upload_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_upload = form.save()
            source = video_upload.video.path

            # Create a unique folder for the upload using the video ID
            upload_folder = os.path.join(settings.MEDIA_ROOT, f'upload_{video_upload.id}')
            os.makedirs(upload_folder, exist_ok=True)

            # Move the original video to the unique folder
            original_video_path = os.path.join(upload_folder, os.path.basename(source))
            os.rename(source, original_video_path)

            # Open video and capture the first frame
            cap = cv2.VideoCapture(original_video_path)
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                # Process the first frame (detect positions and corresponding indices)
                results, first_frame_results, _, _, choose_frame = process_video([first_frame])

                # Save the processed first frame
                choose_frame_filename = f'choose_frame_{video_upload.id}.png'
                choose_frame_path = os.path.join(upload_folder, choose_frame_filename)
                plt.imsave(choose_frame_path, choose_frame)

                # Determine the number of detected humans
                num_humans = len(first_frame_results.cpu().boxes.xyxy.numpy())
                if_proceed = num_humans == 2

                if if_proceed:
                    # If there are exactly two people, proceed automatically
                    tracked_indices = get_largest_two_humans([first_frame_results])[0]
                    return redirect('process_data', video_id=video_upload.id, index1=tracked_indices[0], index2=tracked_indices[1])
                else:
                    # Generate a list of human indices for user input in the template
                    human_indices = list(range(num_humans))

                    # Prompt the user to select the athlete indices
                    context = {
                        'choose_frame_url': os.path.join(settings.MEDIA_URL, f'upload_{video_upload.id}', choose_frame_filename),
                        'video_id': video_upload.id,
                        'num_humans': num_humans,
                        'human_indices': human_indices
                    }
                    return render(request, 'processing/choose_frame.html', context)
            else:
                # Handle video read failure
                return render(request, 'processing/video_upload.html', {'form': form, 'error': 'Failed to read video file.'})
    else:
        form = VideoUploadForm()

    # Return to the upload page
    return render(request, 'processing/video_upload.html', {'form': form})






from .models import VideoUpload
import os
from .video_processing import process_video, extract_data, cut_video
from .video_processing import find_winner, generate_video_with_keypoints_segment, get_frame_intervals, calculate_velocity_acceleration, cut_video_segments  # Import the function from your script
import pickle
from django.conf import settings
from django.http import HttpResponse

from moviepy.editor import VideoFileClip


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plots

import numpy as np
import os
import matplotlib.font_manager as fm

# 生成柱状图
def plot_bar_chart(left_data, right_data, keypoint_labels, ylabel, title, save_path, video_id):
    """Helper function to plot a bar chart comparing left and right fencers with Chinese labels."""
    bar_width = 0.35  # Bar width
    index = np.arange(len(keypoint_labels))  # Bar positions

    # Use Chinese font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    plt.bar(index, left_data, bar_width, label='左方运动员')
    plt.bar(index + bar_width, right_data, bar_width, label='右方运动员')

    # Add Chinese labels
    plt.xlabel('关节点', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(index + bar_width / 2, keypoint_labels)  # Align labels between bars
    plt.legend()

    # Save the plot in the unique upload folder
    output_dir = os.path.join(settings.MEDIA_ROOT, f'upload_{video_id}')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, save_path)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# 生成柱状图2
def generate_bar_plots_for_velocity_acceleration(left_xdata_new_df, right_xdata_new_df, video_id, keypoints=['hand', 14, 16]):
    """Generates and saves bar plots for comparing velocity and acceleration in Chinese."""
    # Keypoint labels
    keypoint_labels = ['手', '前膝盖', '前脚']
    
    # Calculate velocity and acceleration
    left_velocities = []
    left_accelerations = []
    right_velocities = []
    right_accelerations = []

    for keypoint in keypoints:
        left_velocity, left_acceleration = calculate_velocity_acceleration(left_xdata_new_df[keypoint])
        right_velocity, right_acceleration = calculate_velocity_acceleration(right_xdata_new_df[keypoint])

        # Calculate mean velocity and acceleration
        left_velocities.append(abs(left_velocity.mean()))
        left_accelerations.append(abs(left_acceleration.mean()))
        right_velocities.append(abs(right_velocity.mean()))
        right_accelerations.append(abs(right_acceleration.mean()))

    # Create output directory using the video ID
    output_dir = os.path.join(settings.MEDIA_ROOT, f'upload_{video_id}')
    os.makedirs(output_dir, exist_ok=True)

    # Save the velocity comparison plot
    velocity_save_path = os.path.join(output_dir, 'velocity_comparison.png')
    plot_bar_chart(left_velocities, right_velocities, keypoint_labels, '速度', '速度比较', velocity_save_path, video_id)

    # Save the acceleration comparison plot
    acceleration_save_path = os.path.join(output_dir, 'acceleration_comparison.png')
    plot_bar_chart(left_accelerations, right_accelerations, keypoint_labels, '加速度', '加速度比较', acceleration_save_path, video_id)

    # Return the paths of the plots relative to MEDIA_URL
    return {
        'velocity_plot': os.path.join(settings.MEDIA_URL, f'upload_{video_id}', 'velocity_comparison.png'),
        'acceleration_plot': os.path.join(settings.MEDIA_URL, f'upload_{video_id}', 'acceleration_comparison.png')
    }



# 数据处理和判别方法
from django.shortcuts import render
from django.conf import settings
import os
import random


import logging

logger = logging.getLogger(__name__)  # Get the logger

def process_data(request, video_id, index1, index2):
    try:
        # Check if processed data is already in session
        if 'processed_data' in request.session:
            processed_data = request.session['processed_data']
            context = processed_data['context']
            return render(request, 'processing/display_results.html', context)

        print("Starting process_data")
        video_upload = VideoUpload.objects.get(id=video_id)
        
        # Construct the correct path to the video file within its unique folder
        video_filename = os.path.basename(video_upload.video.path)  # Extracts the video filename
        upload_folder = os.path.join(settings.MEDIA_ROOT, f'upload_{video_id}')  # Folder path for this upload
        source = os.path.join(upload_folder, video_filename)  # Full path to the video file
        print(f"Video source: {source}")

        # Check if the video file exists
        if not os.path.exists(source):
            print(f"File does not exist: {source}")
            return render(request, 'processing/display_error.html', {'error_message': 'Video file not found.'})


        if request.method == 'POST':
            index1 = request.POST.get('index1')
            index2 = request.POST.get('index2')
            print(f"Index1: {index1}, Index2: {index2}")

        tracked_indices = [int(index1), int(index2)]

        # Process the video and extract data
        print("Processing video...")
        results, first_frame_results, num_humans, if_proceed, _ = process_video(source)
        print ('Extracting Keypoints')
        left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, checker_list_df, video_angle_df = extract_data(source, results, first_frame_results, tracked_indices)
        print("Video processed.")

        
        # Cut video data
        print("Cutting video data...")
        left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, latest_end, end_frame, longest_segment = cut_video(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, video_angle_df)
        print("Video cut completed.")

        # Generate video with keypoints
        #print("Generating video with keypoints...")
        #video_with_keypoints_path = generate_video_with_keypoints_segment(source, tracked_indices, left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, 'keypoints', video_id, latest_end, end_frame)
        #print("Generated video with keypoints.")
        #video_with_keypoints_url = os.path.join(settings.MEDIA_URL, 'keypoints', os.path.basename(video_with_keypoints_path))

        # Call cut_video_segments to generate keypoint video with just one interval
        keypoint_frame_interval = [(0, len(left_xdata_df) - 5)]

        # Use the existing cut_video_segments function
        keypoint_video_url = cut_video_segments(
            source, 
            keypoint_frame_interval, 
            'keypoint_segments', 
            video_id, 
            left_xdata_df, 
            left_ydata_df, 
            right_xdata_df, 
            right_ydata_df, 
            c
        )

        all_segments_keypoint = []
        # Append the keypoint video to all_segments with the label and start frame
        all_segments_keypoint.append({
            'url': keypoint_video_url[0],  # There will only be one video in this case
            'label': '关键点视频',  # Label for keypoint video
            'start_frame': latest_end  # Start frame can be latest_end
        })

        # Determine winner
        print("Finding winner...")
        final_winner, left_percentage, right_percentage, left_processed_segments, right_processed_segments = find_winner(
            left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, video_angle_df, longest_segment
        )
        print("Winner determined.")
        print (left_processed_segments, right_processed_segments)
        # Adjust percentages
        print("Adjusting percentages...")
        if final_winner == 'left' and right_percentage > left_percentage:
            left_p = round(right_percentage, 2)
            right_p = 100 - left_p
        elif final_winner == 'left' and right_percentage < left_percentage:
            left_p = round(left_percentage, 2)
            right_p = 100 - left_p
        elif final_winner == 'right' and right_percentage < left_percentage:
            left_p = round(right_percentage, 2)
            right_p = 100 - left_p
        elif final_winner == 'right' and right_percentage > left_percentage:
            left_p = round(left_percentage, 2)
            right_p = 100 - left_p
        elif final_winner == 'left' and right_percentage == left_percentage:
            right_p = round(right_percentage, 2)
            left_p = 100 - right_p
        elif final_winner == 'right' and right_percentage == left_percentage:
            left_p = round(left_percentage, 2)
            right_p = 100 - left_p
        elif final_winner == 'sim':
            left_p = 50
            right_p = 50
        print("Percentages adjusted.")

        random_number = round(random.uniform(0.5, 0.9), 2)
        if final_winner == 'left':
            left_p = random_number * 100
            right_p = (1 - random_number) * 100
        else:
            right_p = random_number * 100
            left_p = (1 - random_number) * 100

        # Translate winner to Chinese
        final_winner_chinese = '左边' if final_winner == 'left' else '右边' if final_winner == 'right' else '无法判断'
        print(f"Final winner: {final_winner_chinese}")

        # Case 1: Generate bar plots for velocity and acceleration
        if not left_processed_segments and not right_processed_segments:
            try:
                print("Generating bar plots...")
                plot_urls = generate_bar_plots_for_velocity_acceleration(left_xdata_new_df, right_xdata_new_df, video_id)
                print("Bar plots generated.")

                # Prepare context for case 1 (plots)
                context = {
                    'final_winner': final_winner_chinese,
                    'left_percentage': left_p,
                    'right_percentage': right_p,
                    'plot_urls': plot_urls,  # Case 1
                    'video_with_keypoints_url': all_segments_keypoint,  # Keypoint video
                    'all_segments_sorted': None,  # No segments for case 1
                }

                return render(request, 'processing/display_results.html', context)
            except Exception as e:
                logger.error(f"Error generating bar plots: {e}")
                return render(request, 'processing/display_error.html', {'error_message': '很抱歉，视频无法处理'})

        # Case 2: Output video segments for right of way changes
        else:
            try:
                print("Processing video segments...")
                all_segments = []
                if left_processed_segments:
                    left_frame_intervals = left_processed_segments
                    left_segments_video_url = cut_video_segments(source, left_frame_intervals, 'left_segments', video_id, left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c)
                    for i, url in enumerate(left_segments_video_url):
                        all_segments.append({
                            'url': url,
                            'label': '左方停顿, 后退，或收手，可能造成主动权转换',
                            'start_frame': left_frame_intervals[i][0]  # Using the start frame for sorting
                        })

                if right_processed_segments:
                    right_frame_intervals = right_processed_segments
                    right_segments_video_url = cut_video_segments(source, right_frame_intervals, 'right_segments', video_id, left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c)
                    for i, url in enumerate(right_segments_video_url):
                        all_segments.append({
                            'url': url,
                            'label': '右方停顿，后退，或收手，可能造成主动权转换',
                            'start_frame': right_frame_intervals[i][0]  # Using the start frame for sorting
                        })

                # Sort all segments by the start frame to display them in time order
                all_segments_sorted = sorted(all_segments, key=lambda x: x['start_frame'])
                print(f"All segments sorted by time: {all_segments_sorted}")

                # Prepare context for case 2 (video segments)
                context = {
                    'final_winner': final_winner_chinese,
                    'left_percentage': left_p,
                    'right_percentage': right_p,
                    'plot_urls': None,  # Case 2 (no plots)
                    'video_with_keypoints_url': all_segments_keypoint,
                    #settings.MEDIA_URL + os.path.basename(video_with_keypoints_path),  # Keypoint video
                    'all_segments_sorted': all_segments_sorted,  # Sorted segments with labels
                }

                return render(request, 'processing/display_results.html', context)

            except Exception as e:
                logger.error(f"Error processing video segments: {e}")
                return render(request, 'processing/display_error.html', {'error_message': '很抱歉，视频无法处理'})

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return render(request, 'processing/display_error.html', {'error_message': '很抱歉，视频无法处理'})


    #data = context
    #return JsonResponse(data)





