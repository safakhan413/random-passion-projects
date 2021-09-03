"""
WSGI config for dictionary project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
import django
import sys
sys.path.append("H:\Coding and Hackathons\coding problems\random-passion-projects\python-mysql-dictionary\dictionary")
print (sys.path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dictionary.settings')
django.setup()
application = get_wsgi_application()


