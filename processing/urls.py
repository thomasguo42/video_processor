from django.urls import path
from . import views

urlpatterns = [
    path('', views.video_upload_view, name='video_upload'),
    path('process/<int:video_id>/<int:index1>/<int:index2>/', views.process_data, name='process_data'),
]
