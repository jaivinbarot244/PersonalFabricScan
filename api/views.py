import math
import uuid
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType, Task
import datumaro as dm
from cvat_sdk.core.proxies.projects import Project
from .models import (Basemodel,Fabric,Datasets,Images,YoloModel,Annotator,Tasks,PredictionData)
import json
import zipfile
import os
from datetime import datetime
import csv
import psutil
import shutil
import threading
from ultralytics import YOLO
import torch
from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import csv
import random
import base64
import time
from threading import Thread
import genicam.genapi as ge
from pathlib import Path
from harvesters.core import Harvester
import yaml
from tabulate import tabulate
# Create your views here.
width = int(4096)
height = int(2176)
h = Harvester()
h.add_file("C:/Program Files/Balluff/ImpactAcquire/bin/x64/mvGenTLProducer.cti")
h.update()


predictPath = ""
predictModel = None
predictFolder = ""
os.environ["WANDB_DISABLED"] = "true"
realtimePredict = False
caps = {}
cameraSetting = {}
try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")


try:
    with make_client(
        host="app.cvat.ai", credentials=("basappa", "Tipl@123")
    ) as client:
        CVATClient = client
except:
    CVATClient = None
    print("CVAT Client Not connected")

def createProject(name, labels):
    project_spec = {
        "name": name,
        "labels": labels,
        "target_storage": {"location": "local"},
        "source_storage": {"location": "local"},
    }
    newProject = client.projects.create(spec=project_spec)
    return newProject


def createTask(name, subset, projectId, data):
    task_spec = {
        "name": name,
        "assignee_id": 33017,
        "project_id": projectId,
        "subset": subset,
    }
    task = client.tasks.create_from_data(
        spec=task_spec, resource_type=ResourceType.LOCAL, resources=data
    )
    return task

def sysInfo(request):
    # Get RAM usage
    ram = psutil.virtual_memory()
    ram_usage = ram.percent

    # Get CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)

    # Create a dictionary containing the system information
    data = {"ram": ram_usage, "cpu": cpu_usage}
    # Return the system information as a JSON object
    return HttpResponse(json.dumps(data), content_type="application/json")

def datasetConvertor(dataset: str, output: str):
    """
    This function takes a dataset of format Datumaro 1.0 and converts it to YOLOv8 txt format that can be used to train the model.
    The function takes the dataset path as input and returns the converted dataset path in the required format.
    """
    
    # make a directory to store the converted dataset
    os.makedirs(output, exist_ok=True)
    os.makedirs(output+"/test", exist_ok=True)
    os.makedirs(output+"/train", exist_ok=True)
    os.makedirs(output+"/valid", exist_ok=True)
    os.makedirs(output+"/test/images", exist_ok=True)
    os.makedirs(output+"/train/images", exist_ok=True)
    os.makedirs(output+"/valid/images", exist_ok=True)
    os.makedirs(output+"/test/labels", exist_ok=True)
    os.makedirs(output+"/train/labels", exist_ok=True)
    os.makedirs(output+"/valid/labels", exist_ok=True)
    
    # check if the dataset exists
    if not os.path.exists(dataset):
        print("The dataset folder "+dataset+" does not exist")
    # check if the annotations folder exists
    if not os.path.exists(os.path.join(dataset, "annotations")):
        print("The annotations folder is missing in the dataset folder")
    # check if the images folder exists
    if not os.path.exists(os.path.join(dataset, "images")):
        print("The images folder is missing in the dataset folder")
    
    try:
        # get the test.json file from the dataset/annotations folder
        testPath = os.path.join(dataset, "annotations", "test.json")
        testData = json.loads(open(testPath, "r").read())
    except:
        print("The test.json file is missing in the "+dataset+"/annotations folder")
    try:
        # get the train.json file from the dataset/annotations folder
        trainPath = os.path.join(dataset, "annotations", "train.json")
        trainData = json.loads(open(trainPath, "r").read())
    except:
        print("The train.json file is missing in the "+dataset+"/annotations folder")
    try:
        # get the val.json file from the dataset/annotations folder
        valPath = os.path.join(dataset, "annotations", "val.json")
        valData = json.loads(open(valPath, "r").read())
    except:
        print("The val.json file is missing in the "+dataset+"/annotations folder")
    
    # copy the images to the output folder
    try:
        testImages = os.listdir(os.path.join(dataset, "images", "test"))
        for img in testImages:
            shutil.copy(os.path.join(dataset, "images", "test", img), os.path.join(output, "test", "images", img))
    except:
        print("Error copying the test images to the "+output+" folder")
    try:
        trainImages = os.listdir(os.path.join(dataset, "images", "train"))
        for img in trainImages:
            shutil.copy(os.path.join(dataset, "images", "train", img), os.path.join(output, "train", "images", img))
    except:
        print("Error copying the train images to the "+output+" folder")
    try:
        valImages = os.listdir(os.path.join(dataset, "images", "val"))
        for img in valImages:
            shutil.copy(os.path.join(dataset, "images", "val", img), os.path.join(output, "valid", "images", img))
    except:
        print("Error copying the val images to the "+output+" folder")

    # create the labels for the test images
    try:
        for image in testData["items"]:
            fileName = image["id"]
            tempPath = os.path.join(output, "test", "labels", fileName+".txt")
            with open(tempPath, "w") as labelFile:
                for annotation in image["annotations"]:
                    points = ""
                    count = 0
                    for point in annotation["points"]:
                        if count%2 == 0:
                            point = point/image["image"]["size"][1]
                        else:
                            point = point/image["image"]["size"][0]
                        points += str(point)+" "
                        count += 1
                    labelFile.write(str(annotation["label_id"])+" "+points+"\n")
    except:
        print("Error creating the labels for the test images")
    # create the labels for the train images
    try:
        for image in trainData["items"]:
            fileName = image["id"]
            tempPath = os.path.join(output, "train", "labels", fileName+".txt")
            with open(tempPath, "w") as labelFile:
                for annotation in image["annotations"]:
                    points = ""
                    count = 0
                    for point in annotation["points"]:
                        if count%2 == 0:
                            point = point/image["image"]["size"][1]
                        else:
                            point = point/image["image"]["size"][0]
                        points += str(point)+" "
                        count += 1
                    labelFile.write(str(annotation["label_id"])+" "+points+"\n")
    except:
        print("Error creating the labels for the train images")
    # create the labels for the val images
    try:
        for image in valData["items"]:
            fileName = image["id"]
            tempPath = os.path.join(output, "valid", "labels", fileName+".txt")
            with open(tempPath, "w") as labelFile:
                for annotation in image["annotations"]:
                    points = ""
                    count = 0
                    for point in annotation["points"]:
                        if count%2 == 0:
                            point = point/image["image"]["size"][1]
                        else:
                            point = point/image["image"]["size"][0]
                        points += str(point)+" "
                        count += 1
                    labelFile.write(str(annotation["label_id"])+" "+points+"\n")
                
    except:
        print("Error creating the labels for the val images")

    try:
        # create data.yaml file
        classes = [label["name"] for label in trainData["categories"]["label"]["labels"]]
        currentPath = os.getcwd()
        configData = {
            "train": currentPath+"/"+output+"/train/images",
            "val": currentPath+"/"+output+"/valid/images",
            "test": currentPath+"/"+output+"/test/images",
            "nc": len(classes),
            "names": classes
        }
        yaml_data = yaml.dump(configData)
        tempPath = os.path.join(output, "data.yaml")
        with open(tempPath, 'w') as yaml_file:
            yaml_file.write(yaml_data)
    except:
        print("Error creating the data.yaml file")
    return dataset


@csrf_exempt
def modelTraining(request):
    # If the request method is POST
    if request.method == "POST":
            project = request.POST.get("projectName")
            model = request.POST.get("model",'yolov8n-seg.pt')
            datasetName = request.POST["datasetName"]
            # data = request.POST.get(data,None)
            epochs = request.POST.get('epochs',100)
            # time = request.POST.get(time,None)
            # patience = request.POST.get(patience,50)
            # batch = request.POST.get(batch,16)
            imgsz = request.POST.get('imgsz',640)
            # save = request.POST.get(save,True)
            # save_period = request.POST.get(save_period,-1)
            # cache = request.POST.get(cache,False)
            # device = request.POST.get(device,None)
            # workers = request.POST.get(workers,8)
            # project = request.POST.get(project,None)
            name = request.POST.get('name',None)
            # exist_ok = request.POST.get(exist_ok,False)
            # pretrained = request.POST.get(pretrained,True)
            # optimizer = request.POST.get(optimizer,'auto')
            # verbose = request.POST.get(verbose,False)
            # seed = request.POST.get(seed,0)
            # deterministic = request.POST.get(deterministic,True)
            # single_cls = request.POST.get(single_cls,False)
            # rect = request.POST.get(rect,False)
            # cos_lr = request.POST.get(cos_lr,False)
            # close_mosaic = request.POST.get(close_mosaic,10)
            resume = request.POST.get("resume",False)
            # amp = request.POST.get(amp,True)
            # fraction = request.POST.get(fraction,1.0)
            # profile = request.POST.get(profile,False)
            # freeze = request.POST.get(freeze,None)
            # lr0 = request.POST.get(lr0,0.01)
            # lrf = request.POST.get(lrf,0.01)
            # momentum = request.POST.get(momentum,0.937)
            # weight_decay = request.POST.get(weight_decay,0.0005)
            # warmup_epochs = request.POST.get(warmup_epochs,3.0)
            # warmup_momentum = request.POST.get(warmup_momentum,0.8)
            # warmup_bias_lr = request.POST.get(warmup_bias_lr,0.1)
            # box = request.POST.get(box,7.5)
            # cls = request.POST.get(cls,0.5)
            # dfl = request.POST.get(dfl,1.5)
            # pose = request.POST.get(pose,12.0)
            # kobj = request.POST.get(kobj,2.0)
            # label_smoothing = request.POST.get(label_smoothing,0.0)
            # nbs = request.POST.get(nbs,64)
            # overlap_mask = request.POST.get(overlap_mask,True)
            # mask_ratio = request.POST.get(mask_ratio,4)
            # dropout = request.POST.get(dropout,0.0)
            # val = request.POST.get(val,True)
            # plots = request.POST.get(plots,False)
            baseDir = os.getcwd()
            # Initialize a YOLO model with the specified base model
           
            # print(project,model,epochs,imgsz,name,datasetName)
            # temp = baseDir+"/projects/"+project+"/datasets/"+ datasetName +"/data.yaml"
            # print(temp)
            # Train the model using the specified parameters
            if resume == "true":
                resume = True
                model = YOLO(
                    "projects/" + project + "/models/" + name + "/weights/best.pt"
                )
                name = model
            else:
                baseModel = YOLO("baseModels/" + model)
            baseModel.train(
                data="projects/"+project+"/datasets/"+ datasetName +"/data.yaml",
                # data=baseDir+"/projects/"+project+"/datasets/"+ datasetName+"/data.yaml",
                epochs=int(epochs),
                imgsz=int(imgsz),
                project=baseDir + "/projects/" + project + "/models/",
                name=name,
                batch=-1,
                task="segment",
                resume = resume
            )
            yolomodel = YoloModel()
            yolomodel.fabric = Fabric.objects.filter(fabricName=project).first()
            yolomodel.dataset = Datasets.objects.filter(fabric__fabricName = project,datasetName=datasetName).first()
            yolomodel.baseModel = model
            yolomodel.modelName = name
            yolomodel.epochs = epochs
            yolomodel.imgsz = imgsz
            yolomodel.save()


            # Return a success message as a JSON object
            return HttpResponse(
                json.dumps({"success": "Model Trained "}),
                content_type="application/json",
            )
    #     except Exception as e:
    #         # If an error occurs, return an error message as a JSON object
    #         return HttpResponse(
    #             json.dumps({"error": str(e)}), content_type="application/json"
    #         )
    # else:
    #     # If the request is not a POST request, return an error message as a JSON object
    #     return HttpResponse(
    #         json.dumps({"error": "Invalid Request"}), content_type="application/json"
    #     )

@csrf_exempt
def addNewDataset(request):
    if request.method == 'POST':
        try:
            images = request.FILES.getlist('images')
            datasetName = request.POST.get('datasetName')
            datasetDiscription = request.POST.get('datasetDiscription')
            fabricName = request.POST.get('fabricName')
            fabric  = Fabric.objects.filter(fabricName=fabricName).first()
            print("Details captured",datasetName,datasetDiscription,fabricName)

            imgLen =  len(images)
            datasetTemp  = Datasets()
            datasetTemp.datasetName = datasetName
            datasetTemp.datasetDescription = datasetDiscription
            datasetTemp.fabric = fabric
            datasetTemp.noOfImages = imgLen
            datasetTemp.save()
            print('Dataset Created',datasetTemp.datasetName,datasetTemp.datasetDescription,datasetTemp.fabric,datasetTemp.noOfImages)
            if(imgLen > 0):
                trainCount = math.ceil((imgLen * 70)/ 100)
                testCount = math.ceil((imgLen - trainCount)/2)
                validCount = imgLen - trainCount - testCount
                for i in range(imgLen):
                    if i<=trainCount:
                        Images.objects.create(image=images[i],category='train',dataset=datasetTemp,fabric=fabric)
                    elif i>=trainCount and i<(testCount + trainCount):
                       Images.objects.create(image=images[i],category='test',dataset=datasetTemp,fabric=fabric)
                    elif i>=(testCount + trainCount) and i<=imgLen:
                        Images.objects.create(image=images[i],category='valid',dataset=datasetTemp,fabric=fabric)


            # for image in images:
            #     new_image = Image.objects.create(
            #         image=image,
            #         category=category,
            #         role=role,
            #         dataset=dataset,
            #         fabric=fabric
            #     )

            return HttpResponse(
                json.dumps({"success": "Dataset Created "}),
                content_type="application/json",
            )  # Redirect to success page

        except Exception as e:
            # Handle errors appropriately (e.g., log, display user-friendly message)
            print(e)
            return HttpResponse(
                json.dumps({"Error": "Error creating dataset"}),
                content_type="application/json",
            )

    # Handle GET requests if needed
    else:
         return HttpResponse(
                json.dumps({"msg": "Get Request Failed"}),
                content_type="application/json",
            )  # Redirect to success page

          
        # return render(request, 'modeltraining.html', {'data' : data})
            # return render(request, 'addNewDataset.html', {'data' : data})
    




@csrf_exempt
def addNewFebric(request):
    if request.method == "POST":
        try:
            fabricName = request.POST["fabricName"]
            description = request.POST["fabricDetails"]
            sampleimages = request.FILES.getlist("images")
            # dataset = request.FILES["dataset"]
            fabricType = request.POST["fabricType"]
            color = request.POST["fabricColor"]
            GSM = request.POST["fabricGSM"]
            material = request.POST["fabricMaterial"]
            fabricLabels = request.POST["fabricLabels"]
            datasetimages = request.FILES['dataset']

            print("data collected")
            baseDir = os.getcwd()
            folderPath = baseDir + "/projects/" + fabricName
            fabricData = Fabric.objects.filter(fabricName=fabricName).first()
            print("DATA COLLECTED",fabricName)
            if fabricData == None:
                labels = [
                    {
                        "name": label[0],
                        "attributes": [],
                        "type": "polygon",
                        "color": label[1],
                    }
                    for label in json.loads(fabricLabels)
                ]
                print(labels)
                project = createProject(fabricName, labels)
               
                # print(project)
                os.mkdir(baseDir + "/projects/" + fabricName)
                os.mkdir(folderPath + "/datasets")
                os.mkdir(folderPath + "/images")
                os.mkdir(folderPath + "/models")
                os.mkdir(folderPath + "/predicts")
                os.mkdir(folderPath + "/samples")
                os.mkdir(folderPath + "/tempImages")
                os.mkdir(folderPath + "/output")
                print("folders created")
                newFabric = Fabric()
                newFabric.id = project.id
                newFabric.fabricName = fabricName
                newFabric.fabricDescription = description
                newFabric.GSM = GSM
                newFabric.color = color
                newFabric.fabricType = fabricType
                newFabric.material = material
                newFabric.sampleImages = sampleimages[0]
                newFabric.labels = labels
                newFabric.save()
                print("Fabric")
                for file in sampleimages:
                    with open(baseDir + "/projects/" + fabricName + "/samples/" + file.name, "wb+") as f:
                        f.write(file.read())
                for i in range(len(sampleimages)):
                    Images.objects.create(image=sampleimages[i],category='samples',fabric=newFabric)

                extract_folder = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                datasetName = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

                target_folder = "./projects/" + fabricName + "/datasets/" + datasetName
                with zipfile.ZipFile(datasetimages, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(target_folder, extract_folder))
  

                data = {}
                temp = []
                for path, subdirs, files in os.walk(target_folder):
                    for name in files:
                        temp.append(os.path.join(path,name))
                       
                data["temp"] = temp
                imgLen = len(temp)
                data["count"] = imgLen
                datasetName = 'Dataset1'
                datasetDiscription = 'Dataset1 description'
                datasetTemp  = Datasets()
                datasetTemp.datasetName = datasetName
                datasetTemp.datasetDescription = datasetDiscription
                datasetTemp.fabric = Fabric.objects.filter(fabricName=fabricName).first()
                datasetTemp.noOfImages = imgLen
                datasetTemp.save()
                trainCount = math.ceil((imgLen * 70)/ 100)
                testCount = math.ceil((imgLen - trainCount)/2)
                validCount = imgLen - trainCount - testCount
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName)
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/")
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/")
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/val/")
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/")
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/")
                os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/val/images/")
                testImages = []
                trainImages = []
                validImages = []

                for i in range(imgLen):
                    
                    uuidName = str(uuid.uuid4())
                    if i<=trainCount:
                        tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/" + uuidName+".jpg"
                        os.rename(data["temp"][i],tempPath)
                        trainImages.append(tempPath)
                        tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/" + uuidName+".jpg"
                        Images.objects.create(image=tempPath,category='train',dataset=datasetTemp,fabric=datasetTemp.fabric)
                    elif i>=trainCount and i<(testCount + trainCount):
                        tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/" + uuidName+".jpg"
                        os.rename(data["temp"][i],tempPath)
                        testImages.append(tempPath)
                        tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/" + uuidName+".jpg"
                        Images.objects.create(image=tempPath,category='test',dataset=datasetTemp,fabric=datasetTemp.fabric)
                    elif i>=(testCount + trainCount) and i<=imgLen:
                        tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/val/images/" + uuidName+".jpg"
                        os.rename(data["temp"][i],tempPath)
                        validImages.append(tempPath)
                        tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/val/images/" + uuidName+".jpg"
                        Images.objects.create(image=tempPath,category='valid',dataset=datasetTemp,fabric=datasetTemp.fabric)
                    data["temp"][i] = tempPath


                testTask = createTask(extract_folder + "_test", "test", project.id, testImages)
                trainTask = createTask( extract_folder + "_train", "train", project.id, trainImages)
                validTask = createTask(extract_folder + "_val", "val", project.id, validImages)  
                newTask = Tasks()
                newTask.fabric = newFabric
                newTask.datasetName = datasetTemp
                newTask.tasks = json.dumps(
                    {
                        "test": testTask.id,
                        "train": trainTask.id,
                        "val": validTask.id,
                    }
                )
                newTask.save()
                print("done")
                return HttpResponse(
                    json.dumps({"success": "Fabric Created successfully"}),
                    content_type="application/json",
                )
            else:
                return HttpResponse(
                    json.dumps({"error": "Fabric Already exists."}),
                    content_type="application/json",
                )
        except Exception as e:
            print("ERror:" , e)
            return HttpResponse(
                json.dumps({"error": str(e)}),
                content_type="application/json",
            )
    else:
        return HttpResponse(
            json.dumps({"erroe": "Invalid Request"}),
            content_type="application/json",
        )


def getFebricDetails(request):
    try:
       
        fabricName = request.GET["fabricName"]
        imagesPath = []
        fabric = Fabric.objects.filter(fabricName=fabricName).first()
        images = Images.objects.filter(fabric=fabric).all()[:10]
        for image in images:
            imagesPath.append(image.image.url[1:])
        return HttpResponse(
            json.dumps({"description": fabric.fabricDescription, "images": imagesPath}),
            content_type="application/json",
        )
    except Exception as e:
        # If an error occurs, return an empty description and image list as a JSON object
        return HttpResponse(
            json.dumps({"description": "", "images": []}),
            content_type="application/json",
        )



def imageGallery(request):
    try:
        # Get the project and dataset names from the request
        fabric = request.GET["project"]
        dataset = request.GET["dataset"]
        # Initialize an empty dictionary to store the image paths
        data = {}
        # Define the paths to the test, train, and validation image folders
        # testPath = "projects/" + project + "/datasets/" + dataset + "/test/images/"
        # trainPath = "projects/" + project + "/datasets/" + dataset + "/train/images/"
        # validPath = "projects/" + project + "/datasets/" + dataset + "/valid/images/"
        # Initialize empty lists for the test, train, and validation image paths
        data["test"] = []
        data["train"] = []
        data["valid"] = []
        fabric  = Fabric.objects.filter(fabricName=fabric).first()
        dataset = Datasets.objects.filter(datasetName=dataset, fabric=fabric).first()
        images = Images.objects.filter(fabric=fabric, dataset=dataset).all()

        for image in images:
            if image.category == "test":
                data["test"].append(image.image.url[1:])
            elif image.category == "train":
                data["train"].append(image.image.url[1:])
            elif image.category == "valid":
                data["valid"].append(image.image.url[1:])    
        data["project"] = fabric.fabricName
        data["dataset"] = dataset.datasetName
        # Return the dictionary as a JSON object
        return HttpResponse(json.dumps(data), content_type="application/json")
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )


def basemodelFecther(request):
    try:
        # Initialize an empty list to store the base model folders
        project = request.GET.get("project",None)
        data = {"baseModel": [], "projectModel": []}
        
        if project is not None:
            yoloModels = YoloModel.objects.filter(fabric__fabricName=project)
            print("project:- ", project)
            for yoloModel in yoloModels:
                data['projectModel'] .append(yoloModel.modelName)
                
        basemodels = Basemodel.objects.all()
        for basemodel in basemodels:
            data['baseModel'] .append(basemodel.modelName)
        return HttpResponse(
            json.dumps({"data": data}), content_type="application/json"
        )
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )


def modelFecther(request):
    try:
        project = request.GET["project"]
        folders = []
        fabric = Fabric.objects.filter(fabricName=project).first()
        yolomodel = YoloModel.objects.filter(fabric = fabric)
        for yolo in yolomodel:
            folders.append(yolo.modelName)
        return HttpResponse(
            json.dumps({"data": folders}), content_type="application/json"
        )
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )

def modelDetails(request):
    try:
        # Get the project and model names from the request
        fabricName = request.GET["fabricName"]
        modelName = request.GET["modelName"]
        # Define the path to the results CSV file

        csvFile = "projects/" + fabricName + "/models/" + modelName + "/results.csv"
        # Initialize an empty list to store the model details
        data = []
        # Open the CSV file and read the rows
        with open(csvFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            # For each row, extract the relevant data and append it to the list
            for row in csv_reader:
                temp = [
                    row[0].strip().replace(" ", ""),
                    row[5].strip().replace(" ", ""),
                    row[6].strip().replace(" ", ""),
                    row[7].strip().replace(" ", ""),
                    row[8].strip().replace(" ", ""),
                ]
                data.append(temp)
        # Return the model details as a JSON object
        return HttpResponse(
            json.dumps({"data": data[1:]}), content_type="application/json"
        )
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )



def datasetFecther(request):
    try:
        project = request.GET["project"]
        folders = []
        # for folder in os.listdir("projects/" + project + "/datasets"):
        #     if os.path.isdir("projects/" + project + "/datasets/" + folder):
        #         folders.append(folder)
        datasets = Datasets.objects.filter(fabric__fabricName=project)
        for dataset in datasets:
            folders.append(dataset.datasetName)
            print(dataset.datasetName)
        return HttpResponse(
            json.dumps({"data": folders}), content_type="application/json"
        )
    except Exception as e:
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )



def febricFecther(request):
    try:
        # Initialize an empty list to store the project folders
        folders = []
        fabrics = Fabric.objects.all()
        for fabric in fabrics:
            folders.append(fabric.fabricName)
        return HttpResponse(
            json.dumps({"data": folders}), content_type="application/json"
        )
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )


@csrf_exempt
def demo(request):
    fabricName = request.POST["fabricName"]
    
    datasetimages = request.FILES['datasetimages']
    extract_folder = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    target_folder = "./projects/" + fabricName + "/tempImages/"
    with zipfile.ZipFile(datasetimages, "r") as zip_ref:
        zip_ref.extractall(os.path.join(target_folder, extract_folder))
  

    data = {}
    temp = []
    for path, subdirs, files in os.walk(target_folder):
        for name in files:
            temp.append(os.path.join(path,name))
           
    data["temp"] = temp
    imgLen = len(temp)
    data["count"] = imgLen
    print(data["temp"][1])
      
    datasetTemp  = Datasets()
    datasetTemp.datasetName = "dummy data test"
    datasetTemp.datasetDescription = "dummy data"
    datasetTemp.fabric = Fabric.objects.filter(fabricName=fabricName).first()
    datasetTemp.noOfImages = imgLen
    datasetTemp.save()
    trainCount = math.ceil((imgLen * 70)/ 100)
    testCount = math.ceil((imgLen - trainCount)/2)
    validCount = imgLen - trainCount - testCount
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName)
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/")
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/")
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/valid/")
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/")
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/")
    os.mkdir("./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/valid/images/")
    
    for i in range(imgLen):
        
        uuidName = str(uuid.uuid4())
        if i<=trainCount:
            tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/" + uuidName+".jpg"
            os.rename(data["temp"][i],tempPath)
            tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/train/images/" + uuidName+".jpg"
            Images.objects.create(image=tempPath,category='train',dataset=datasetTemp,fabric=datasetTemp.fabric)
        elif i>=trainCount and i<(testCount + trainCount):
            tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/" + uuidName+".jpg"
            os.rename(data["temp"][i],tempPath)
            tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/test/images/" + uuidName+".jpg"
            Images.objects.create(image=tempPath,category='test',dataset=datasetTemp,fabric=datasetTemp.fabric)
        elif i>=(testCount + trainCount) and i<=imgLen:
            tempPath = "./projects/"+fabricName+"/datasets/"+datasetTemp.datasetName+"/valid/images/" + uuidName+".jpg"
            os.rename(data["temp"][i],tempPath)
            tempPath = fabricName+"/datasets/"+datasetTemp.datasetName+"/valid/images/" + uuidName+".jpg"
            Images.objects.create(image=tempPath,category='valid',dataset=datasetTemp,fabric=datasetTemp.fabric)
        data["temp"][i] = tempPath

    # imagesTest = os.listdir(target_folder + extract_folder)
    # # )
    # # imagesTrain = os.listdir(
    # #     target_folder + extract_folder + "/train/images/"
    # # )
    # # imagesValid = os.listdir(
    # #     target_folder + extract_folder + "/valid/images/"
    # # )
    # images = []
    # for i in range(len(imagesTest)):
    #     images.append(
    #         target_folder
    #         + extract_folder
    #         + "/test/images/"
    #         + imagesTest[i]
    #     )
    
    return HttpResponse(
            json.dumps({"data": data}), content_type="application/json"
        )
    
def projectDetails(request):
    try:
        data = []

        fabrics = Fabric.objects.all()
        # for folder in os.listdir("projects"):
        for fabric in fabrics:
                datasets = Datasets.objects.filter(fabric=fabric)
                datasetCount = datasets.count()
                modelCount = 0
                yolomodel = YoloModel.objects.filter(fabric=fabric)
                modelCount = yolomodel.count()
                # for item in os.listdir("projects/" + fabric.fabricName + "/models"):
                #     item_path = os.path.join(
                #         "projects/" + fabric.fabricName + "/models", item
                #     )
                #     if os.path.isdir(item_path):
                #         modelCount += 1

                sampleImages = []
                images = Images.objects.filter(fabric=fabric)
                for image in images:
                    if image.category == "samples":
                        sampleImages.append(image.image.url[1:])
                temp = {
                    "id": fabric.id,
                    "name": fabric.fabricName,
                    "datasetCount": datasetCount,
                    "modelCount": modelCount,
                    "sampleImages": sampleImages,
                }
                data.append(temp)
        # Return the list of project folders as a JSON object
        return HttpResponse(json.dumps({"data": data}), content_type="application/json")
    except Exception as e:
        # If an error occurs, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": str(e)}), content_type="application/json"
        )
def AnnotatorDetails(request):
    try:
        annotators = Annotator.objects.all()
        annotatorList = []
        for annotator in annotators:
            annotatorList.append([annotator.annotatorId, annotator.annotatorName])
        return HttpResponse(
            json.dumps({"data": annotatorList}),
            content_type="application/json",
        )
    except Exception as e:
        return HttpResponse(
            json.dumps({"error": str(e)}),
            content_type="application/json",
        )

@csrf_exempt
def projectDetailsFetch(request):
    projectId = request.GET.get("projectID")
    fabric = Fabric.objects.filter(id=projectId).first()
    data = {}
    fname = fabric.fabricName
    data["fabricName"] = fname
    data["fabricDescription"] = fabric.fabricDescription
    data["GSM"] = fabric.GSM
    data["color"] = fabric.color
    data["fabricType"] = fabric.fabricType
    data["material"] = fabric.material
    data["labels"] = fabric.labels
    data["images"] = []

    # list all folder names in projects/fabric.fabricName/datasets
    datasets = Datasets.objects.filter(fabric=fabric)
    # datasets = os.listdir("projects/" + fname + "/datasets")
    datasetDetail = []
    for dataset in datasets:
        temp = {}
        temp["datasetName"] = dataset.datasetName
        # trainPath = "projects/" + fname + "/datasets/" + dataset + "/train/images/"
        temp["images"] = []
        images = Images.objects.filter(dataset = dataset)
        try: 
            for image in images:
                # print(image.image.url)
                image = (image.image.url[1:])
                temp["images"].append(image)
            trainImages = Images.objects.filter(dataset = dataset,category='train')
            temp["trainCount"] = trainImages.count()
            validImages = Images.objects.filter(dataset = dataset,category='valid')
            temp["validCount"] = validImages.count()
            testImages = Images.objects.filter(dataset = dataset,category='test')
            temp["testCount"] = testImages.count()
            data["images"].append(temp["images"][0])
            data["images"].append(temp["images"][1])

            datasetDetail.append(temp)
        except Exception as e:
            data["images"]  = []
    data["datasetDetail"] = datasetDetail
    folders = []
    models = YoloModel.objects.filter(fabric = fabric)
    for model in models:
        folders.append(model.modelName)
    data["models"] = folders

    return HttpResponse(json.dumps({"data": data}), content_type="application/json")




@csrf_exempt
def extractFolder(request):
    # Check if the request method is POST and if the "dataset" file was uploaded
    if request.method == "POST":
        if request.FILES.get("dataset", False):
            # Check if a project name was provided
            projectName = request.POST["projectName"]
            # Get the uploaded file and the project name from the request
            dataset = request.FILES["dataset"]
            # Check if a dataset name was provided
            datasetName = request.POST.get("datasetName",'')
            if projectName != "":
                if request.POST.get("datasetName", False):
                    print(datasetName)
                    # create a new temp folder
                    extract_folder = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    temp_folder = "temp/" + projectName + "/"
                    target_folder = (
                        "projects/" + projectName + "/datasets/" + datasetName + "/"
                    )

                    project = Fabric.objects.filter(fabricName=projectName).first()
                    # Extract the files from the zip file to the target folder
                    with zipfile.ZipFile(dataset, "r") as zip_ref:
                        zip_ref.extractall(os.path.join(temp_folder, extract_folder))
                    print("Data Extracted 123")

                    
                    # get the list of images in the test folder
                    dataset = dm.Dataset.import_from(
                        temp_folder + extract_folder + "/" ,"open_images"
                    )

                    dataset.export(
                        temp_folder + extract_folder + "/yolo/",
                        "yolo_ultralytics",
                        save_media=True,
                    )
                    print("Data Converted 123")
                    # copy all folders and files from the temp_folder + extract_folder + "/yolo/labels" to the target folder
                    for file in os.listdir(
                        temp_folder + extract_folder + "/yolo/labels/test/"
                    ):
                        with open(
                            temp_folder + extract_folder + "/yolo/labels/test/" + file,
                            "rb",
                        ) as f:
                            with open(
                                target_folder + "test/labels/" + file, "wb+"
                            ) as target:
                                target.write(f.read())
                    for file in os.listdir(
                        temp_folder + extract_folder + "/yolo/labels/train/"
                    ):
                        with open(
                            temp_folder + extract_folder + "/yolo/labels/train/" + file,
                            "rb",
                        ) as f:
                            with open(
                                target_folder + "train/labels/" + file, "wb+"
                            ) as target:
                                target.write(f.read())
                    for file in os.listdir(
                        temp_folder + extract_folder + "/yolo/labels/val/"
                    ):
                        with open(
                            temp_folder + extract_folder + "/yolo/labels/val/" + file,
                            "rb",
                        ) as f:
                            with open(
                                target_folder + "valid/labels/" + file, "wb+"
                            ) as target:
                                target.write(f.read())
                    with open(
                        temp_folder + extract_folder + "/yolo/data.yaml",
                        "r",
                    ) as f:
                        lines = f.readlines()
                        lines[0] = "train: ../train/images\n"
                        lines[1] = "val: ../valid/images\n"
                        lines[2] = "test: ../test/images\n"
                        with open(
                            target_folder + "data.yaml",
                            "w",
                        ) as target:
                            target.writelines(lines)
                    print("Data Imported")
                    data = {}
                    temp = []
                    for path, subdirs, files in os.walk(target_folder):
                        for name in files:
                            temp.append(os.path.join(path,name))
                    data["temp"] = temp
                    imgLen = len(temp)
                    newdataset  = Datasets()
                    newdataset.datasetName = datasetName
                    newdataset.datasetDescription = "Created at" + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    newdataset.fabric = project
                    newdataset.noOfImages = imgLen
                    newdataset.save()
                    
                    trainCount = math.ceil((imgLen * 70)/ 100)
                    testCount = math.ceil((imgLen - trainCount)/2)
                    for i in range(imgLen):
        
                        uuidName = str(uuid.uuid4())
                        if i<=trainCount:
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/train/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/train/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='train',dataset=newdataset,fabric=newdataset.fabric)
                        elif i>=trainCount and i<(testCount + trainCount):
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/test/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/test/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='test',dataset=newdataset,fabric=newdataset.fabric)
                        elif i>=(testCount + trainCount) and i<=imgLen:
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/valid/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/valid/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='valid',dataset=newdataset,fabric=newdataset.fabric)
                        data["temp"][i] = tempPath




                    return HttpResponse(
                        json.dumps({"success": "File extracted successfully!"}),
                        content_type="application/json",
                    )
                else:
                    # Create a folder name based on the current date and time
                    extract_folder = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    temp_folder = "temp/" + projectName + "/"
                    target_folder = "projects/" + projectName + "/datasets/"

                    project = Fabric.objects.filter(fabricName=projectName).first()
                    # Extract the files from the zip file to the target folder
                    with zipfile.ZipFile(dataset, "r") as zip_ref:
                        zip_ref.extractall(os.path.join(temp_folder, extract_folder))
                    print("Data Extracted")

                    datasetConvertor((temp_folder+extract_folder),(target_folder+extract_folder))
                    # # get the list of images in the test folder
                    # dataset = dm.Dataset.import_from(
                    #     temp_folder + extract_folder + "/" ,"open_images"
                    # )

                    # # dataset.export(
                    # #     temp_folder + extract_folder + "/yolo/",
                    # #     "yolo_ultralytics",
                    # #     save_media=True,
                    # # )
                    # dataset.export(
                    #     temp_folder + extract_folder + "/yolo/",
                    #     "yolo_ultralytics",
                    #     save_media=True,
                    # )
                    print("Data Converted")
                    
                    # # copy all folders and files from the temp_folder + extract_folder + "/yolo/labels" to the target folder
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/images/test/"):
                    #     with open( temp_folder + extract_folder + "/yolo/images/test/" + file, "rb",) as f:
                    #         with open(target_folder + "test/images/" + file, "wb+") as target:
                    #                     target.write(f.read())
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/images/train/"):
                    #     with open(temp_folder + extract_folder + "/yolo/images/train/" + file,"rb",) as f:
                    #         with open(target_folder + "train/images/" + file, "wb+") as target:
                    #             target.write(f.read())
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/images/val/"):
                    #         with open(temp_folder + extract_folder + "/yolo/images/val/" + file,"rb",) as f:
                    #             with open(target_folder + "valid/images/" + file, "wb+") as target:
                    #                 target.write(f.read())
                    # #Labels
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/labels/test/"):
                    #     with open( temp_folder + extract_folder + "/yolo/labels/test/" + file, "rb",) as f:
                    #         with open(target_folder + "test/labels/" + file, "wb+") as target:
                    #                     target.write(f.read())
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/labels/train/"):
                    #     with open(temp_folder + extract_folder + "/yolo/labels/train/" + file,"rb",) as f:
                    #         with open(target_folder + "train/labels/" + file, "wb+") as target:
                    #             target.write(f.read())
                    # for file in os.listdir(temp_folder + extract_folder + "/yolo/labels/val/"):
                    #         with open(temp_folder + extract_folder + "/yolo/labels/val/" + file,"rb",) as f:
                    #             with open(target_folder + "valid/labels/" + file, "wb+") as target:
                    #                 target.write(f.read())
                    # baseDir = os.getcwd()
                    # with open(temp_folder + extract_folder + "/yolo/data.yaml","rb") as f:
                    #     tempdata = f.read()
                    #     print(tempdata)
                    #     print(target_folder + "data.yaml")
                    #     with open(target_folder + "data.yaml","wb") as target:
                    #         target.write(tempdata)
                    #         print("data.yaml created")

                    print("Data Imported")
                    data = {}
                    temp = []
                    for path, subdirs, files in os.walk(target_folder):
                        for name in files:
                            temp.append(os.path.join(path,name))
                    data["temp"] = temp
                    imgLen = len(temp)
                    print(imgLen)
                    newdataset  = Datasets()
                    newdataset.datasetName = extract_folder
                    newdataset.datasetDescription = "Created at" + extract_folder
                    newdataset.fabric = project
                    newdataset.noOfImages = imgLen
                    newdataset.save()
                    trainCount = math.ceil((imgLen * 70)/ 100)
                    testCount = math.ceil((imgLen - trainCount)/2)
                    for i in range(imgLen):
        
                        uuidName = str(uuid.uuid4())
                        if i<=trainCount:
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/train/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/train/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='train',dataset=newdataset,fabric=newdataset.fabric)
                           
                        elif i>=trainCount and i<(testCount + trainCount):
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/test/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/test/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='test',dataset=newdataset,fabric=newdataset.fabric)
                        elif i>=(testCount + trainCount) and i<=imgLen:
                            tempPath = "./projects/"+projectName+"/datasets/"+newdataset.datasetName+"/valid/images/" + uuidName+".jpg"
                            os.rename(data["temp"][i],tempPath)
                            tempPath = projectName+"/datasets/"+newdataset.datasetName+"/valid/images/" + uuidName+".jpg"
                            Images.objects.create(image=tempPath,category='valid',dataset=newdataset,fabric=newdataset.fabric)
                        data["temp"][i] = tempPath
                    # Return a success message if the extraction was successful
                    return HttpResponse(
                        json.dumps({"success": "File extracted successfully!"}),
                        content_type="application/json",
                    )
            else:
                # Return an error message if no project name was provided
                return HttpResponse(
                    json.dumps({"error": "No project name found"}),
                    content_type="application/json",
                )
        else:
            # Return an error message if no file was found in the request
            return HttpResponse(
                json.dumps({"error": "No file found"}), content_type="application/json"
            )

@csrf_exempt
def modelPrediction(request):
    if request.method == "POST":

        # Initialize variables for the prediction path, model, and folder
        global predictPath, predictModel, predictFolder, caps
        results = []
        # Get the project name, model name, and input source from the request
        project = request.POST["projectName"]
        modelName = request.POST["modelName"]
        inputSource = request.POST["inputSource"]
        # Check if the input source is an uploaded image
        if inputSource == "image":
            # If it is, save the image to the project's images directory
            image = request.FILES["image"]
            # save the image to the project's images directory
            with open("projects/" + project + "/images/" + image.name, "wb+") as f:
                f.write(image.read())
            # image.save("projects/" + project + "/images/" + image.filename)
            # Check if the current model is different from the previous prediction model
            if ("./projects/" + project + "/models/" + modelName + "/weights/best.pt"!= predictPath):
                # If it is, update the prediction path, model, and folder
                predictPath = (
                    "projects/" + project + "/models/" + modelName + "/weights/best.pt"
                )
                predictModel = YOLO(
                    "projects/" + project + "/models/" + modelName + "/weights/best.pt"
                )
                predictFolder = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            try:
                # Predict the objects in the image using the YOLO model
                results = predictModel(
                    "projects/" + project + "/images/" + image.name,
                    retina_masks=True,
                    iou=0.1,
                    conf=0.1,
                    imgsz=640,
                    project="projects/" + project + "/predicts/",
                    name=predictFolder,
                    save=True,
                    show_conf=False,
                    show_labels=False,
                    show_boxes=False,
                )
                names = predictModel.names
                data = []
                for result in results:
                    orignalPath = result.path
                    path = (
                        "projects/"
                        + project
                        + "/predicts/"
                        + predictFolder
                        + "/"
                        + image.name
                    )
                    imageCv = cv2.imread(path)
                    for box in result.boxes:
                        boxCor = box.xyxy.tolist()[0]
                        centerX = boxCor[0] + ((boxCor[2] - boxCor[0]) / 2)
                        centerY = boxCor[1] + ((boxCor[3] - boxCor[1]) / 2)
                        confident = box.conf.tolist()[0]
                        # plot the point on the image
                        imageHeight, imageWidth, _ = imageCv.shape
                        imageCv = cv2.circle(
                            imageCv,
                            (int(centerX), int(centerY)),
                            10,
                            (0, 255, 255),
                            10,
                        )
                        imageCv = cv2.rectangle(
                            imageCv,
                            (int(boxCor[0]), int(boxCor[1])),
                            (int(boxCor[2]), int(boxCor[3])),
                            (0, 255, 255),
                            1,
                        )
                        imageCv = cv2.putText(
                            imageCv,
                            str(confident)[:4],
                            (int(boxCor[0]), int(boxCor[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        
                        xy = box.xyxy.tolist()[0]
                        area_pixels = (xy[2] - xy[0]) * (xy[3] - xy[1])
                        pixel_to_inch = 0.01  # Replace with your actual conversion factor
                        area_inches = area_pixels * pixel_to_inch ** 2
                        points = 0
                        if area_inches > 0 and area_inches < 3:
                            points = 1
                        elif area_inches > 3 and area_inches < 6:
                            points = 2
                        elif area_inches > 6 and area_inches < 9:
                            points = 3
                        elif area_inches > 9:
                            points = 4  
                        temp = []
                        temp.append(names[int(box.cls)])
                        temp.append(centerX)
                        temp.append(centerY)
                        temp.append(area_inches)
                        temp.append(points)
                        data.append(temp)
                        print(data)
                        newPredict = PredictionData()
                        newPredict.fabric = Fabric.objects.filter(
                            fabricName=project
                        ).first()
                        newPredict.centroid = str(centerX)[:4] + "," + str(centerY)[:4]
                        newPredict.boundingBox = (
                            str(boxCor[0])[:4]
                            + ","
                            + str(boxCor[1])[:4]
                            + ","
                            + str(boxCor[2])[:4]
                            + ","
                            + str(boxCor[3])[:4]
                        )
                        newPredict.imageRaw = (
                            "projects/" + project + "/images/" + image.name
                        )
                        newPredict.imageAnnotated = path
                        newPredict.confidence = confident
                        newPredict.save()
                    outputstr = tabulate(data, headers=["Name","centerX", "centerY","area_inches","points"],tablefmt='orgtbl')  
                    print(outputstr) 
                    # save the image
                    cv2.imwrite(path, imageCv)
                return HttpResponse(
                    json.dumps(
                        {
                            "path": project
                            + "/predicts/"
                            + predictFolder
                            + "/"
                            + image.name,
                            "results": outputstr.replace(' ',' ')
                        }
                    ),
                    content_type="application/json",
                )
            except Exception as e:
                return HttpResponse(
                    json.dumps(
                        {
                            "error": str(e),
                            "status": "error",
                        }
                    ),
                    content_type="application/json",
                )
        else:
            # If no input source is provided, return an error message as a JSON object
            return HttpResponse(
                json.dumps({"error": "No Input Source Provided"}),
                content_type="application/json",
            )
    else:
        # If an invalid request is made, return an error message as a JSON object
        return HttpResponse(
            json.dumps({"error": "Invalid Request"}), content_type="application/json"
        )


def webcam(cameraIndex, project, modelName):
    global predictPath, predictModel, predictFolder, caps, realtimePredict, cameraSetting
    cameraSetting["Camera" + str(cameraIndex)] = {
        "exposure": 25,
        "saturation": 1.0,
        "contrast": 1.0,
    }
    # global outputfile 
    
        
    try:
        caps["Camera" + str(cameraIndex)] = cv2.VideoCapture(int(cameraIndex))
    except:
        try:
            caps["Camera" + str(cameraIndex)] = cv2.VideoCapture(
                "/dev/video" + str(cameraIndex)
            )
        except:
            pass
    filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fields = ['name', 'X', 'Y', 'area','point']
    with open("projects/" + project + "/output/" + modelName + filename + ".csv", 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames = fields)
        csvwriter.writeheader()
    while True:
        success, image = caps["Camera" + str(cameraIndex)].read()
        if success:
            image_np = np.array(image)
            image_np = cv2.convertScaleAbs(
                image_np,
                alpha=cameraSetting["Camera" + str(cameraIndex)]["contrast"],
                beta=cameraSetting["Camera" + str(cameraIndex)]["exposure"],
            )
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = (
                hsv[:, :, 1] * cameraSetting["Camera" + str(cameraIndex)]["saturation"]
            )
            image_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            data = []
            if realtimePredict:
                if ("projects/" + project + "/models/" + modelName + "/weights/best.pt"!= predictPath):
                    # If it is, update the prediction path, model, and folder
                    predictPath = ("projects/"+ project+ "/models/"+ modelName+ "/weights/best.pt")
                    predictModel = YOLO("projects/"+ project+ "/models/"+ modelName+ "/weights/best.pt")
                results = predictModel(source=image_np,iou=0.1,conf=0.3,imgsz=256,save=False,)# show_conf=True,# show_labels=True,# show_boxes=True,)
                names = predictModel.names
                
                count = 0
                data = []
                for result in results:
                    count = count + 1
                    for box in result.boxes:
                        boxCor = box.xyxy.tolist()[0]
                        centerX = boxCor[0] + ((boxCor[2] - boxCor[0]) / 2)
                        centerY = boxCor[1] + ((boxCor[3] - boxCor[1]) / 2)
                        confident = box.conf.tolist()[0]
                        # plot the point on the image
                        imageHeight, imageWidth, _ = image_np.shape
                        image_np = cv2.circle(
                            image_np,
                            (int(centerX), int(centerY)),
                            10,
                            (0, 255, 255),
                            10,
                        )
                        image_np = cv2.rectangle(
                            image_np,
                            (int(boxCor[0]), int(boxCor[1])),
                            (int(boxCor[2]), int(boxCor[3])),
                            (0, 255, 255),
                            1,
                        )
                        image_np = cv2.putText(
                            image_np,
                            str(confident)[:4],
                            (int(boxCor[0]), int(boxCor[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        xy = box.xyxy.tolist()[0]
                        area_pixels = (xy[2] - xy[0]) * (xy[3] - xy[1])
                        pixel_to_inch = 0.01  # Replace with your actual conversion factor
                        area_inches = area_pixels * pixel_to_inch ** 2
                        points = 0
                        if area_inches > 0 and area_inches < 3:
                            points = 1
                        elif area_inches > 3 and area_inches < 6:
                            points = 2
                        elif area_inches > 6 and area_inches < 9:
                            points = 3
                        elif area_inches > 9:
                            points = 4  
                        
                        
                        print("Area area_inches = ",area_inches)
                        print("Names = ",names[int(box.cls)])
                        print("Center points = ",centerX,centerY)
                        print("Points = ",points)
                        print("id:= ",count)
                        temp = []
                        temp.append(names[int(box.cls)])
                        temp.append(centerX)
                        temp.append(centerY)
                        temp.append(area_inches)
                        temp.append(points)
                        data.append(temp)
            # filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            # fields = ['name', 'X', 'Y', 'area','point']
            with open("projects/" + project + "/output/" + modelName + filename+ ".csv", 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in data:
                    csvwriter.writerow(row)
            # convert the image to base64
            _, img_encoded = cv2.imencode(".jpg", np.array(image_np))
            img_bytes = img_encoded.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + img_bytes + b"\r\n"
            )
        else:
            caps["Camera" + str(cameraIndex)].release()

def mainCam(cameraIndex, project, modelName):
    global predictPath, predictModel, predictFolder, caps, realtimePredict, h, cameraSetting
    global filename 
    filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fields = ['name', 'X', 'Y', 'area','point']
    with open("projects/" + project + "/output/" + modelName + filename + ".csv", 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames = fields)
        csvwriter.writeheader()
    index = 0
    cameraSetting["Camera" + str(cameraIndex)] = {
        "exposure": 25,
        "saturation": 1.0,
        "contrast": 1.0,
    }
    for i in h.device_info_list:
        if i.serial_number == cameraIndex:
            break
        index = index + 1
    try:
        caps["Camera" + str(cameraIndex)] = h.create(index)
    except:
        h.update()
        caps["Camera" + str(cameraIndex)] = h.create(index)
    caps["Camera" + str(cameraIndex)].remote_device.node_map.Width.value = width
    caps["Camera" + str(cameraIndex)].remote_device.node_map.Height.value = height
    caps["Camera" + str(cameraIndex)].remote_device.node_map.PixelFormat.value = (
        "BayerRG8"
    )
    caps["Camera" + str(cameraIndex)].remote_device.node_map.ChunkSelector.value = (
        "ExposureTime"
    )
    caps["Camera" + str(cameraIndex)].remote_device.node_map.ExposureTime.set_value(
        8000.0
    )
    caps["Camera" + str(cameraIndex)].start()
    filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fields = ['name', 'X', 'Y', 'area','point']
    with open("projects/" + project + "/output/" + modelName + filename + ".csv", 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames = fields)
        csvwriter.writeheader()
    while True:
        with caps["Camera" + str(cameraIndex)].fetch() as buffer:
            component = buffer.payload.components[0]
            image_np = component.data.reshape(height, width)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BayerRG2RGB)
            image_np = cv2.resize(image_np, (int(width / 4), int(height / 4)))
            # increase the exposure
            image_np = cv2.convertScaleAbs(
                image_np,
                alpha=cameraSetting["Camera" + str(cameraIndex)]["contrast"],
                beta=cameraSetting["Camera" + str(cameraIndex)]["exposure"],
            )
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = (
                hsv[:, :, 1] * cameraSetting["Camera" + str(cameraIndex)]["saturation"]
            )
            image_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            if realtimePredict:
                if (
                    "projects/" + project + "/models/" + modelName + "/weights/best.pt"
                    != predictPath
                ):
                    # If it is, update the prediction path, model, and folder
                    predictPath = (
                        "projects/"
                        + project
                        + "/models/"
                        + modelName
                        + "/weights/best.pt"
                    )
                    predictModel = YOLO(
                        "projects/"
                        + project
                        + "/models/"
                        + modelName
                        + "/weights/best.pt"
                    )
                    names = predictModel.names
                results = predictModel(
                    source=image_np,
                    iou=0.1,
                    conf=0.1,
                    imgsz=640,
                    save=False,
                    # show_conf=True,
                    # show_labels=True,
                    # show_boxes=True,
                )
                data = []
                for result in results:
                    for box in result.boxes:
                        boxCor = box.xyxy.tolist()[0]
                        centerX = boxCor[0] + ((boxCor[2] - boxCor[0]) / 2)
                        centerY = boxCor[1] + ((boxCor[3] - boxCor[1]) / 2)
                        confident = box.conf.tolist()[0]
                        # plot the point on the image
                        imageHeight, imageWidth, _ = image_np.shape
                        image_np = cv2.circle(
                            image_np,
                            (int(centerX), int(centerY)),
                            10,
                            (0, 255, 255),
                            10,
                        )
                        image_np = cv2.rectangle(
                            image_np,
                            (int(boxCor[0]), int(boxCor[1])),
                            (int(boxCor[2]), int(boxCor[3])),
                            (0, 255, 255),
                            1,
                        )
                        image_np = cv2.putText(
                            image_np,
                            str(confident)[:4],
                            (int(boxCor[0]), int(boxCor[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        xy = box.xyxy.tolist()[0]
                        area_pixels = (xy[2] - xy[0]) * (xy[3] - xy[1])
                        pixel_to_inch = 0.01  # Replace with your actual conversion factor
                        area_inches = area_pixels * pixel_to_inch ** 2
                        points = 0
                        if area_inches > 0 and area_inches < 3:
                            points = 1
                        elif area_inches > 3 and area_inches < 6:
                            points = 2
                        elif area_inches > 6 and area_inches < 9:
                            points = 3
                        elif area_inches > 9:
                            points = 4  
                        
                        
                        print("Area area_inches = ",area_inches)
                        print("Names = ",names[int(box.cls)])
                        print("Center points = ",centerX,centerY)
                        print("Points = ",points)
                        
                        temp = []
                        temp.append(names[int(box.cls)])
                        temp.append(centerX)
                        temp.append(centerY)
                        temp.append(area_inches)
                        temp.append(points)
                        data.append(temp)
            # filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            # fields = ['name', 'X', 'Y', 'area','point']
            with open("projects/" + project + "/output/" + modelName + filename + ".csv", 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in data:
                    csvwriter.writerow(row)
                      
            # convert the image to base64
            _, img_encoded = cv2.imencode(".jpg", np.array(image_np))
            img_bytes = img_encoded.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + img_bytes + b"\r\n"
            )
    pass


@csrf_exempt
def realtimePrediction(request):
    cameraIndex = request.GET["cameraIndex"]
    project = request.GET["project"]
    modelName = request.GET["modelName"]

    # read the camera using opencv
    try:
        return StreamingHttpResponse(
            webcam(cameraIndex, project, modelName),
            content_type="multipart/x-mixed-replace;boundary=frame",
        )
    except Exception as e:
        image = open("static/error.png", "rb").read()
        return HttpResponse(
            image,
            content_type="image/png",
        )


def toggleRealtimePrediction(request):
    global realtimePredict
    try:
        if realtimePredict:
            realtimePredict = False
        else:
            realtimePredict = True
        return HttpResponse(
            json.dumps(
                {
                    "status": "success",
                    "data": realtimePredict,
                }
            ),
            content_type="application/json",
        )
    except Exception as e:
        return HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                }
            ),
            content_type="application/json",
        )

def engageCamera(request, index):
    try:
        global caps
        try:
            index = int(index)
            caps["Camera" + str(index)] = cv2.VideoCapture(index)
        except:
            try:
                index = int(index)
                caps["Camera" + str(index)] = cv2.VideoCapture(
                    "/dev/video" + str(index)
                )
            except:
                pass
    except Exception as e:
        pass
    return HttpResponse(
        json.dumps(
            {
                "status": "success",
            }
        ),
        content_type="application/json",
    )






def disengageCamera(request, index):
    global caps
    try:
        caps["Camera" + str(index)].stop()
        caps["Camera" + str(index)].destroy()
    except Exception as e:
        try:
            caps["Camera" + str(index)].release()
        except:
            pass

    return HttpResponse(
        json.dumps(
            {
                "status": "success",
            }
        ),
        content_type="application/json",
    )

def toggleRealtimePrediction(request):
    global realtimePredict
    try:
        if realtimePredict:
            realtimePredict = False
        else:
            realtimePredict = True
        return HttpResponse(
            json.dumps(
                {
                    "status": "success",
                    "data": realtimePredict,
                }
            ),
            content_type="application/json",
        )
    except Exception as e:
        return HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                }
            ),
            content_type="application/json",
        )





@csrf_exempt
def cameraListFetcher(request):
    global h
    try:
        cameras = []
        count = 0
        for i in h.device_info_list:
            cameras.append([i.serial_number, i.display_name])
            count += 1
        if count > 0:
            return HttpResponse(
                json.dumps({"data": cameras}), content_type="application/json"
            )
        else:
            raise Exception("No Camera Found")
    except Exception as e:
        print(e)
        # get all camera using cv2
        cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                pass
            else:
                id = i
                name = "Camera " + str(i)
                cameras.append([id, name])
            cap.release()
        return HttpResponse(
            json.dumps({"data": cameras}), content_type="application/json"
        )

@csrf_exempt
def CameraSetting(request):
    global cameraSetting
    if request.method == "POST":
        try:
            cameraIndex = request.POST["cameraIndex"]
            exposure = request.POST.get("exposure", None)
            saturation = request.POST.get("saturation", None)
            contrast = request.POST.get("contrast", None)
            if exposure is not None:
                cameraSetting["Camera" + str(cameraIndex)]["exposure"] = int(exposure)
            if saturation is not None:
                cameraSetting["Camera" + str(cameraIndex)]["saturation"] = float(
                    saturation
                )
            if contrast is not None:
                cameraSetting["Camera" + str(cameraIndex)]["contrast"] = float(contrast)
            return HttpResponse(
                json.dumps(
                    {
                        "status": "success",
                    }
                ),
                content_type="application/json",
            )
        except Exception as e:
            return HttpResponse(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(e),
                    }
                ),
                content_type="application/json",
            )
    else:
        return HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "error": "Invalid Request",
                }
            ),
            content_type="application/json",
        )






