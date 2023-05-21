from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.core.files.storage import default_storage

from pathlib import Path

import image_app.config as cfg
import random
from models.experimental import attempt_load
import numpy as np
import os, uuid, glob, cv2
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, check_imshow, \
    scale_coords, xyxy2xywh, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier

import torch
import torch.backends.cudnn as cudnn


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
str_uuid = uuid.uuid4()  # The UUID for image uploading

def detect(weights, image, img_size=640, device='', conf_threshold=0.3, iou_threshold=0.4, classes=None, 
           max_det=100, line_thickness=3, save_img=True, save_loc='runs/detect', object_count={}):
    
    pred_augment = True
    save_img = True
    view_img = True
    save_txt = False
    object_labels = []
    webcam = image.isnumeric() or image.endswith('.txt') or image.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(save_loc), exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(str(device))
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
       modelc = load_classifier(name='resnet101', n=2)  # initialize
       modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(image, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(image, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=pred_augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=pred_augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=classes, agnostic=True)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #object_count[model.names[int(c)]] += int(n.item())

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                    #if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        object_labels.append(names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)

        output_image = im0

        return output_image, object_count, object_labels

def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploaded_pics/*.*')), recursive=True)
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploaded_pics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

# Create your models here.
class ImagePage(Page):

    template = "image_app/image.html"
    max_count = 2
    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)
    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]


    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"]= []
        context["my_result_file_names"]=[]
        context["my_staticSet_names"]= []
        context["my_lines"]= []
        context["detected_objects"] = []
        return context

    def serve(self, request):
        # Get parameters from python config file
        weights_dir = cfg.weights['directory']
        weights_fn = cfg.weights['file_name']
        weights_file_path = str(os.path.join(ROOT, weights_dir, weights_fn))

        data_dir = cfg.data['directory']
        data_fn = cfg.data['file_name']
        data_file_path = str(os.path.join(ROOT, data_dir, data_fn))

        # Return model and detect function
        emptyButtonFlag = False

        if request.POST.get('start')=="":
            context = self.reset_context(request)
            print(request.POST.get('start'))
            print("Start selected")
            fileroot = os.path.join(settings.MEDIA_ROOT, 'uploaded_pics')
            res_f_root = os.path.join(settings.MEDIA_ROOT, 'Result')
            with open(Path(f'{settings.MEDIA_ROOT}/uploaded_pics/img_list.txt'), 'r') as f:
                image_files = f.readlines()

            # If images exist in the img_list.txt file run model for those images
            if len(image_files)>=0:
                for file in image_files:
                    filename = file.split('/')[-1]
                    filepath = os.path.join(fileroot, filename.strip()) # strip required to remove any carriage returns
                    img = cv2.imread(filepath.strip())
                    confidence_threshold = cfg.thresholds['confidence']
                    iou_threshold = cfg.thresholds['iou']
                    num_classes = cfg.classes
                    line_thickness = cfg.plot['line_thickness']

                    # Run model and detect obejcts/classes in image
                    output_image, object_count, object_labels = detect(weights_file_path,
                                                                       filepath, 
                                                                       device='',
                                                                       conf_threshold=confidence_threshold,
                                                                       iou_threshold=iou_threshold,
                                                                       line_thickness=line_thickness,
                                                                       save_loc=res_f_root)
                    
                    # Write output image to results folder
                    fn = filename.split('.')[:-1][0]
                    r_filename = f'result_{fn}.jpg'
                    cv2.imwrite(str(os.path.join(res_f_root, r_filename)), output_image)
                    r_media_filepath = Path(f"{settings.MEDIA_URL}Result/{r_filename}")
                    print(r_media_filepath)
                    with open(Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'), 'a') as f:
                        f.write(str(r_media_filepath))
                        f.write("\n")
                    context["my_uploaded_file_names"].append(str(f'{str(file)}'))
                    context["my_result_file_names"].append(str(f'{str(r_media_filepath)}'))
                    context["detected_objects"].append(object_labels)
            return render(request, "image_app/image.html", context)

        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
            context = self.reset_context(request)
            context["my_uploaded_file_names"] = []
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploaded_pics/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploaded_pics/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploaded_pics/img_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")

                context["my_uploaded_file_names"].append(str(f'{str(filename)}'))
            return render(request, "image_app/image.html", context)
        context = self.reset_context(request)
        reset()
        return render(request, "image_app/image.html", {'page': self})
