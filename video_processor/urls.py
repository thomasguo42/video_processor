"""
URL configuration for video_processor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin  # Add this line to import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from processing import views

urlpatterns = [
    path('admin/', admin.site.urls),  # This line uses the admin module
    path('', include('processing.urls')),
    path('upload/', views.video_upload_view, name='video_upload'),
    path('process/<int:video_id>/<int:index1>/<int:index2>/', views.process_data, name='process_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

