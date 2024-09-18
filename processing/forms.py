from django import forms
from .models import VideoUpload
from django.core.exceptions import ValidationError
import mimetypes
import cv2
import tempfile

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['video']

    def clean_video(self):
        video = self.cleaned_data.get('video')

        # 检查视频文件的 MIME 类型
        if video:
            mime_type, _ = mimetypes.guess_type(video.name)
            if not mime_type or not mime_type.startswith('video'):
                raise ValidationError('仅允许上传视频文件。')

        # 使用临时文件保存视频并获取其路径
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
            for chunk in video.chunks():
                temp_video.write(chunk)
            temp_video.flush()

            # 使用 OpenCV 读取视频
            cap = cv2.VideoCapture(temp_video.name)
            if not cap.isOpened():
                raise ValidationError('无法打开视频文件。')

            # 获取视频的帧率和总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # 计算视频时长（秒）
            duration = frame_count / fps

            # 检查视频长度是否超过 20 秒
            if duration > 20:
                raise ValidationError('视频长度不能超过20秒，请剪切后重新上传。')

            cap.release()

        return video
