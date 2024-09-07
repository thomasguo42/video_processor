from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from .forms import VideoUploadForm
from .models import VideoUpload
import os
from .video_processing import process_video, extract_data, get_largest_two_humans, cut_video
import matplotlib.pyplot as plt
import cv2



def video_upload_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_upload = form.save()
            source = video_upload.video.path

            # Open video and capture the first frame
            cap = cv2.VideoCapture(source)
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                # Process only the first frame
                results, first_frame_results, _, _, choose_frame = process_video([first_frame])

                # Save the frame with bounding boxes
                choose_frame_filename = f'choose_frame_{video_upload.id}.png'  # Save a unique file for each upload
                choose_frame_path = os.path.join(settings.MEDIA_ROOT, choose_frame_filename)
                plt.imsave(choose_frame_path, choose_frame)

                # Determine how many humans are detected
                num_humans = len(first_frame_results.cpu().boxes.xyxy.numpy())
                if_proceed = num_humans == 2

                if if_proceed:
                    # Automatically proceed if two humans are detected
                    tracked_indices = get_largest_two_humans([first_frame_results])[0]
                    return redirect('process_data', video_id=video_upload.id, index1=tracked_indices[0], index2=tracked_indices[1])
                else:
                    # Ask user for input if more than 2 humans are detected
                    context = {
                        'choose_frame_url': settings.MEDIA_URL + choose_frame_filename,  # Provide full URL
                        'video_id': video_upload.id,
                        'num_humans': num_humans
                    }
                    return render(request, 'processing/choose_frame.html', context)
            else:
                # Handle video reading failure
                return render(request, 'processing/video_upload.html', {'form': form, 'error': 'Failed to read video file.'})
    else:
        form = VideoUploadForm()

    # Render the upload form
    return render(request, 'processing/video_upload.html', {'form': form})




from .models import VideoUpload
import os
from .video_processing import process_video, extract_data, cut_video
from .video_processing import find_winner, generate_video_with_keypoints  # Import the function from your script
import pickle
from django.conf import settings


def process_data(request, video_id, index1, index2):
    video_upload = VideoUpload.objects.get(id=video_id)
    source = video_upload.video.path

    # Get indices from POST request
    if request.method == 'POST':
        index1 = request.POST.get('index1')
        index2 = request.POST.get('index2')

    tracked_indices = [int(index1), int(index2)]

    # Process the entire video and get the dataframes
    results, first_frame_results, num_humans, if_proceed, _ = process_video(source)
    left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, checker_list_df, video_angle_df, _ = extract_data(source, results, first_frame_results, tracked_indices)

    video_with_keypoints_path = generate_video_with_keypoints(results, source, tracked_indices)

    # Cut the video and generate new processed dataframes
    left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, latest_end, end_frame = cut_video(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, video_angle_df)

    # Call the find_winner function with processed dataframes
    final_winner, left_percentage, right_percentage, left_processed_segments, right_processed_segments = find_winner(
        left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, video_angle_df
    )
    if final_winner == 'left' and right_percentage > left_percentage:
        left_p = right_percentage
        right_p = left_percentage
    elif final_winner == 'left' and right_percentage < left_percentage:
        left_p = left_percentage
        right_p = right_percentage
    elif final_winner == 'right' and right_percentage < left_percentage:
        left_p = right_percentage
        right_p = left_percentage
    elif final_winner == 'right' and right_percentage > left_percentage:
        left_p = left_percentage
        right_p = right_percentage

    video_url = settings.MEDIA_URL + os.path.basename(video_with_keypoints_path)

    # Prepare context for the template
    context = {
        'final_winner': final_winner,
        'left_percentage': left_p,
        'right_percentage': right_p,
        'video_with_keypoints_url': video_url,
    }

    return render(request, 'processing/display_results.html', context)





