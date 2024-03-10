from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
   path("sysInfo/", views.sysInfo, name="sysInfo"),
   path("training/", views.modelTraining, name="modelTraining"),
   path("addNewDataset/", views.addNewDataset, name="addNewDataset"),
   path("addNewFebric/", views.addNewFebric, name="addNewFebric"),
   path("projectFecther/", views.febricFecther, name="febricFecther"),
   path("getFebricDetails/", views.getFebricDetails, name="getFebricDetails"),
   path("imageGallery/", views.imageGallery, name="imageGallery"),
   path("basemodelFecther/", views.basemodelFecther, name="basemodelFecther"),
   path("modelFecther/", views.modelFecther, name="modelFecther"),
   path("datasetFecther/", views.datasetFecther, name="datasetFecther"),
   path("projectDetailsFetch/", views.projectDetailsFetch, name="projectDetailsFetch"),
   path("modelDetails/", views.modelDetails, name="modelDetails"),
   path("demo/", views.demo, name="demo"),
   path("prediction/", views.modelPrediction, name="modelPrediction"),
   path('projectDetails/',views.projectDetails,name='projectDetails'),
   path("AnnotatorDetails/", views.AnnotatorDetails, name="AnnotatorDetails"),
   path("extractFolder/", views.extractFolder, name="extractFolder"),
   path("realtimePrediction/", views.realtimePrediction, name="realtimePrediction"),
   path("engageCamera/<index>", views.engageCamera, name="engageCamera"),
   path("disengageCamera/<index>", views.disengageCamera, name="disengageCamera"),
   path("toggleRealtimePrediction/",views.toggleRealtimePrediction,name="toggleRealtimePrediction"),
   path("cameraListFetcher/", views.cameraListFetcher, name="cameraListFetcher"),
   path("CameraSetting/", views.CameraSetting, name="CameraSetting"),



# urlpatterns = [
#     path("sysInfo/", views.sysInfo, name="sysInfo"),
#     path("extractFolder/", views.extractFolder, name="extractFolder"),
#     path("projectFecther/", views.projectFecther, name="projectFecther"),
#     path("projectDetails/", views.projectDetails, name="projectDetails"),
#     path("datasetFecther/", views.datasetFecther, name="datasetFecther"),
#     path("modelFecther/", views.modelFecther, name="modelFecther"),
#     path("basemodelFecther/", views.basemodelFecther, name="basemodelFecther"),
#     path("imageGallery/", views.imageGallery, name="imageGallery"),
#     path("training/", views.modelTraining, name="modelTraining"),
#     path("prediction/", views.modelPrediction, name="modelPrediction"),
#     path("realtimePrediction/", views.realtimePrediction, name="realtimePrediction"),
#     path("modelDetails/", views.modelDetails, name="modelDetails"),
#     path("getFebricDetails/", views.getFebricDetails, name="getFebricDetails"),
#     path("addNewFebric/", views.addNewFebric, name="addNewFebric"),
#     path("taskDetails/", views.taskDetails, name="taskDetails"),
#     path("AnnotatorDetails/", views.AnnotatorDetails, name="AnnotatorDetails"),
#     path("AnnotatorAllocation/", views.AnnotatorAllocation, name="AnnotatorAllocation"),
#     path("projectDetailsFetch/", views.projectDetailsFetch, name="projectDetailsFetch"),
#     path("engageCamera/<index>", views.engageCamera, name="engageCamera"),
#     path("disengageCamera/<index>", views.disengageCamera, name="disengageCamera"),
#     path(
#         "toggleRealtimePrediction/",
#         views.toggleRealtimePrediction,
#         name="toggleRealtimePrediction",
#     ),
#     path("cameraListFetcher/", views.cameraListFetcher, name="cameraListFetcher"),
#     path("CameraSetting/", views.CameraSetting, name="CameraSetting"),

]
 

