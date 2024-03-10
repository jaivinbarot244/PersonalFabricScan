from django.contrib import admin
from .models import (Basemodel,Fabric,Datasets,Images,YoloModel,Annotator,Tasks,PredictionData)
from import_export.admin import ExportActionMixin

# Register your models here.
class BasemodelAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ("modelName", "modelType")
    search_fields = ["modelName", "modelType"]


class FabricAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "fabricName",
        "fabricDescription",
        "GSM",
        "color",
        "fabricType",
        "material",
        
    )
    search_fields = [
        
        "fabricName",
        "fabricDescription",
        "GSM",
        "color",
        "fabricType",
        "material",
    ]

class DatasetsAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "datasetName",
        "datasetDescription",
        "fabric",
   
    )
    search_fields = ["datasetName", "datasetDescription"]

class ImagesAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "datasetPhoto",
        "category",  
        "fabric",
       
    )
    search_fields = ["image", "fabric","category"]


class YoloModelAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ("id","modelName","fabric","dataset")
    search_fields = ["modelName"]


class AnnotatorAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ("annotatorId", "annotatorName", "annotatorEmail")
    search_fields = ["annotatorId", "annotatorName", "annotatorEmail"]

class TasksAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ("id", "fabric", "tasks")
    search_fields = ["fabric"]

class PredictionDataAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = (
        "fabric",
        "time",
        "centroid",
        "boundingBox",
        "image1",
        "image2",
        "confidence",
    )
    search_fields = ["fabric", "time", "centroid", "boundingBox", "confidence"]



admin.site.register(Basemodel, BasemodelAdmin)
admin.site.register(Fabric, FabricAdmin)
admin.site.register(Datasets, DatasetsAdmin)
admin.site.register(Images, ImagesAdmin)
admin.site.register(YoloModel, YoloModelAdmin)
admin.site.register(Annotator, AnnotatorAdmin)
admin.site.register(Tasks, TasksAdmin)
admin.site.register(PredictionData, PredictionDataAdmin)
