"""
WSGI config for edittodo_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
import sys
sys.path.append("H:\Coding and Hackathons\coding problems\random-passion-projects\python-mysql-dictionary\edittodo_project")
print (sys.path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'edittodo_project.settings')

application = get_wsgi_application()
