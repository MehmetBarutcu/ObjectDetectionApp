import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#For SORT tracking
import skimage
from sort import *

import time

import json
import datetime

#............................... Tracker Functions ............................

""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#area = [(552,505),(560,660),(1211,552),(1127,490)]
#area = [(201,831),(285,965),(1691,481),(1545,413)]

def replace_turkish_characters(text):
    turkish_chars = "çÇğĞıİöÖşŞüÜ"
    english_chars = "cCgGiIoOsSuU"
    table = str.maketrans(turkish_chars, english_chars)
    return text.translate(table)

def draw_counting_area(area,img,color):
    cv2.polylines(img, [area], True, color, 2)

"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

"""Simple function that adds fixed color depending on the class"""
def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def count_vehicles(bbox, identities, names, categories, area, v_count, idx, vehicle_set, last_n_sec_det, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        result = int(cv2.pointPolygonTest(np.array(area,np.int32),(center[0],center[1]),False))
        if result >=0 and identities[i] not in vehicle_set:
            last_n_sec_det.add(identities[i])
            vehicle_set.add(identities[i])
            v_count[idx][names[int(categories[i])]] +=1
        #cv2.pointPolygonTest(np.array(start_area,np.int32),(int(midpoint_x),int(midpoint_y)),False)
    return v_count, vehicle_set, last_n_sec_det
            
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, categories=None, identities=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        midpoint_x = x1+((x2-x1)/2)
        midpoint_y = y1+((y2-y1)/2)
        cv2.circle(img, (int(midpoint_x),int(midpoint_y)), 2, (0,255,0), 2)
        cv2.putText(img, f'({midpoint_x},{midpoint_y})', (int(midpoint_x),int(midpoint_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 1)
        cv2.putText(img, f'{names[int(categories[i])]}', (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.90, color, 2, cv2.LINE_AA)
       
    return img
#..............................................................................


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, minute = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, float(opt.interval)
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 

    #......................... 
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    files = []

    t0 = time.time()

    # Initilize vehicle set
    # Tracking vehicles
    vehicle_set = set()
    last_n_sec_det = set()
    for path, img, im0s, vid_cap in dataset:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        f_name = path.split('\\')[-1].replace('.mp4','')
        f_name = replace_turkish_characters(f_name)

        # Last frame for interval
        max_frame = fps*60*minute

        frame = dataset.frame

        if f_name not in files: # New video
            # İnitilize dictionary for inter-minutes
            files.append(f_name)

            # İnitilize list for all results in video
            results = []

            # İnitilize counter for interval
            interval = 1

            # Number of polygones in video
            npol = len(coordinates[f_name])

            # İnitilize for counting dictionary
            v_count = [{name:0 for name in names} for _ in range(npol)]

            # Counter for last n sec detections
            last_n = 5*fps
                
        # Reset vehicle set with 5 sec buffer
        if interval % 3 == 0 and frame == ((max_frame*(interval-1))+(5*fps)):
            vehicle_set = last_n_sec_det.copy()

        if frame == last_n:
            last_n_sec_det = set()
            last_n *= 5 
        
        if frame > max_frame*interval:
            start_time = datetime.timedelta(minutes=(interval-1)*minute)
            end_time = datetime.timedelta(minutes=(interval)*minute)

            start_time = datetime.datetime.strptime(str(start_time),'%H:%M:%S').strftime('%H:%M:%S')
            end_time = datetime.datetime.strptime(str(end_time),'%H:%M:%S').strftime('%H:%M:%S')

            obj = {
                 'start_time':start_time,
                 'end_time': end_time,
                 'result': v_count
                  }
            # Append results
            results.append(obj)

            # Number of polygones in video
            npol = len(coordinates[f_name])

            # Reset vehicle count for next interval
            v_count = [{name:0 for name in names} for _ in range(npol)]

            # Increment interval
            interval +=1

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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

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

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
               
                print('Tracked Detections : '+str(len(tracked_dets)))
                # draw boxes for visualization
                #print(tracked_dets)
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    #print(categories)
                    color = (0,255,0)
                    for idx,area in enumerate(coordinates[f_name].values()):
                        v_count, vehicle_set, last_n_sec_det = count_vehicles(bbox_xyxy,identities,names,categories,area,v_count,idx, vehicle_set, last_n_sec_det)
                        draw_counting_area(np.array(area,np.int32),im0,color)
                    draw_boxes(im0, bbox_xyxy, categories, identities, names)
                    #print('Bbox xy count : '+str(len(bbox_xyxy)))
                    #print('vehicle count : '+str(v_count))                
                #........................................................
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            thickness = 1
            fontScale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (100,570)
            for i,polygon in enumerate(v_count):
                y = org[1] - (i*20)
                cv2.putText(im0, f'Poly={i} '+str(polygon), (org[0],y), font, fontScale, color, thickness, cv2.LINE_AA)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if frame == dataset.nframes:
            start_time = datetime.timedelta(minutes=(interval-1)*minute)
            end_time = datetime.timedelta(minutes=(interval)*minute)

            start_time = datetime.datetime.strptime(str(start_time),'%H:%M:%S').strftime('%H:%M:%S')
            end_time = datetime.datetime.strptime(str(end_time),'%H:%M:%S').strftime('%H:%M:%S')          
            obj = {
                 'start_time':start_time,
                 'end_time':end_time,
                 'result': v_count
                  }
            results.append(obj)

            base = Path(path.split('\\')[-2],increment_path=True, exist_ok = True)
            jsonpath = base / (f_name + ".json")
            base.mkdir(exist_ok=True)
            jsonpath.write_text(json.dumps({'results': results}))

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--interval', default=5, help= 'minute interval for counting')
    parser.add_argument('--configfile', help= 'config file for coordinates')

    opt = parser.parse_args()

    # Read config file
    with open(opt.configfile) as f:
        coordinates = json.load(f)

    # Convert list of coordinates to tuple
    def convert_list_to_tuple(config):
        for fname in config.keys():
            for cname,c in config[fname].items():
                a = list(map(tuple, c))
                config[fname][cname] = a
        return config

    coordinates = convert_list_to_tuple(coordinates)
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()