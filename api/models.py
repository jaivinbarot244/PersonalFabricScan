import datetime
from django.db import models
from django.utils.html import mark_safe
import os
# Create your models here.


def sample_path(instance, filename):
    return os.path.join("projects", instance.fabricName, "samples", filename)

def image_path(instance, filename):
    if instance.category == "samples":
        return os.path.join( instance.fabric.fabricName, "samples", filename)
    else:    
        return os.path.join( instance.fabric.fabricName, "datasets", instance.dataset.datasetName,instance.category, "images", filename)


class Basemodel(models.Model):
    modelName = models.CharField(max_length=100, default="", blank=True)
    modelType = models.CharField(max_length=100, default="", blank=True)
    timestamp = models.DateTimeField(default = datetime.datetime.now(),blank=True)

    def __str__(self):
        return self.modelName
    
class Fabric(models.Model):
    fabricName = models.CharField(max_length=100, default="", blank=True)
    fabricDescription = models.CharField(max_length=1000, default="", blank=True)
    GSM = models.CharField(max_length=100, default="", blank=True)
    color = models.CharField(max_length=100, default="", blank=True)
    fabricType = models.CharField(max_length=100, default="", blank=True)
    material = models.CharField(max_length=100, default="", blank=True)
    sampleImages = models.ImageField(upload_to=sample_path, default="", blank=True)
    labels = models.JSONField(default=dict, blank=True)
    timeStamp = models.DateTimeField(default = datetime.datetime.now(), blank=True)

    def __str__(self):
        return self.fabricName

    # def fabricPhoto(self):
    #     return mark_safe(
    #         f'<img src="/static{self.sampleImages.name.replace("projects","")}" width = "50"/>'
    #     )

class Datasets(models.Model):
    datasetName = models.CharField(max_length=100, default="", blank=True)
    datasetDescription = models.CharField(max_length=1000, default="", blank=True)
    # sampleImages = models.ImageField(upload_to=sample_path, default="", blank=True)
    fabric = models.ForeignKey(Fabric, on_delete=models.CASCADE)
    cvatDeatials = models.JSONField(default=dict, blank=True)
    noOfImages = models.IntegerField(default=0, blank=True)
    modelCount = models.IntegerField(default=0, blank=True)
    timeStamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.datasetName
    
  

    
class Images(models.Model):
    category = models.CharField(max_length=100, default="", blank=True)
    image = models.ImageField(upload_to=image_path, default="", blank=True)
    role = models.CharField(max_length=100, default="not annotated", blank=True)
    dataset  = models.ForeignKey(Datasets, on_delete=models.CASCADE,null = True,blank = True)
    fabric = models.ForeignKey(Fabric, on_delete=models.CASCADE,null = True,blank = True) 
    timeStamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.category

    def datasetPhoto(self):
        return mark_safe(
            f'<img src="/static/{self.image.name.replace("projects/","")}" width = "50"/>'
        )



class YoloModel(models.Model):
    fabric = models.ForeignKey(Fabric, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    baseModel = models.ForeignKey(Basemodel, on_delete=models.CASCADE)
    modelName = models.CharField(max_length=100, default="", blank=True)
    epochs = models.IntegerField(default=10)
    imgsz = models.IntegerField(default=640)

    def __str__(self):
        return self.modelName





class Annotator(models.Model):
    annotatorId = models.CharField(max_length=100, default="", blank=True)
    annotatorName = models.CharField(max_length=100, default="", blank=True)
    annotatorEmail = models.CharField(max_length=100, default="", blank=True)

    def __str__(self):
        return self.annotatorName


class Tasks(models.Model):
    fabric = models.ForeignKey(Fabric, on_delete=models.CASCADE)
    datasetName = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    tasks = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return self.datasetName



class PredictionData(models.Model):
    fabric = models.ForeignKey(Fabric, on_delete=models.CASCADE)
    time = models.DateTimeField(auto_now_add=True)
    centroid = models.CharField(max_length=100, default="", blank=True)
    boundingBox = models.CharField(max_length=100, default="", blank=True)
    imageRaw = models.CharField(max_length=500, default="", blank=True)
    imageAnnotated = models.CharField(max_length=500, default="", blank=True)
    confidence = models.FloatField(default=0.0, blank=True)

    def image1(self):
        return mark_safe(
            f'<img src="/static{self.imageRaw.replace("projects","")}" width = "50"/>'
        )

    def image2(self):
        return mark_safe(
            f'<img src="/static{self.imageAnnotated.replace("projects","")}" width = "50"/>'
        )
