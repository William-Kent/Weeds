"""in_the_weeds URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.conf import settings
from django.contrib import admin
from django.urls import include, path

from wagtail.admin import urls as wt_admin_urls
from wagtail import urls as wt_urls
from wagtail.documents import urls as wt_doc_urls

from search import views as sv
from image_app import views as iv

urlpatterns = [
    path('django-admin/', admin.site.urls),

    path('admin/', include(wt_admin_urls)),
    path('documents/', include(wt_doc_urls)),

    path('search/', sv.search, name='search'),
    path('img/', iv.ImageView.as_view(), name='img'),

    path('', include(wt_urls)),

]

if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)