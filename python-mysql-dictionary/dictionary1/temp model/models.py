from django.db import models

# Create your models here.

class ThesuarusItem(models.Model):
    word = models.TextField()
    meaning = models.TextField()
