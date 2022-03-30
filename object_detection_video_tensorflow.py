import cv2
import numpy
import tensorflow as tf

def read_image_and_resize(img, desired_width, desired_height):
    resized_img = cv2.resize(img,(desired_width,desired_height))
    return resized_img

def convert_image_to_tensor(img):
    image_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor , 0)
    return image_tensor

def load_pretrained_model(model_name):
    #model = tf.keras.models.load_model('faster_rcnn_inception_resnet_v2_640x640_1')
    #model = tensorflow_hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    
    #model = tf.keras.models.load_model('efficientdet_lite2_detection_1')
    #model = tensorflow_hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
    
    # option 1 download the model file from https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1
    # download to same folder as this notebook, untar it
    
    model = tf.keras.models.load_model(model_name)
    
    #option 2 load the model from tensorflow hub 
    #model = tensorflow_hub.load(model_name)
    
    return model

#download 2017 Train/Val annotations [241MB] from https://cocodataset.org/#download
# unzip
import json
def read_coco_class_labels():
    f = open('annotations/instances_train2017.json')
    json_data = json.loads(f.read())
    classes = json_data['categories']
    #print ('classes',classes)
    class_dict={}
    for c in classes:
        class_dict[c['id']]= c['name']
    print ('class_dict',class_dict)
    return class_dict


class_dict = read_coco_class_labels()


cap = cv2.VideoCapture(0)

cnt = 0

while(True):
    ret, frame = cap.read()

    
    model_name = 'efficientdet_lite2_detection_1'
    #model_name = 'faster_rcnn_inception_resnet_v2_640x640_1'
    #if model_name == 'https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1':
    if model_name == 'efficientdet_lite2_detection_1':
        desired_width = 448
        desired_height = 448
    if model_name == 'faster_rcnn_inception_resnet_v2_640x640_1':
        desired_width = 640
        desired_height = 640
    resized_img = read_image_and_resize(frame, desired_width, desired_height)
    image_tensor = convert_image_to_tensor(resized_img)

    if cnt == 0:
        predictor = load_pretrained_model(model_name)
    if model_name == 'efficientdet_lite2_detection_1':
        boxes, scores, classes_id, num_detections = predictor(image_tensor)

    if model_name == 'faster_rcnn_inception_resnet_v2_640x640_1':
        prediction = predictor(image_tensor)
        boxes = prediction['detection_boxes']
        scores = prediction['detection_scores']
        classes_id = prediction['detection_classes']
        num_detections = prediction['num_detections']
    #boxes, scores, classes_id, num_detections = predict_classes_and_boxes(model_name, image_tensor, cnt)
    
    cnt += 1
    #class_dict = read_coco_class_labels()
    top_class_label= class_dict[classes_id[0][0].numpy()]
    top_class_label
    bounding_boxes =boxes[0]
    top_bounding_box = bounding_boxes[0].numpy()

    top_bounding_box_y_min = int(top_bounding_box[0])
    top_bounding_box_x_min = int(top_bounding_box[1])
    top_bounding_box_y_max = int(top_bounding_box[2])
    top_bounding_box_x_max = int(top_bounding_box[3])
    print ('top_bounding_box_y_min',top_bounding_box_y_min)
    print ('top_bounding_box_x_min',top_bounding_box_x_min)
    print ('top_bounding_box_y_max',top_bounding_box_y_max)
    print ('top_bounding_box_x_max',top_bounding_box_x_max)
    top_score=scores[0][0]
    top_score = top_score.numpy()
    resized_img = read_image_and_resize(frame, desired_width, desired_height)
    image_object_box =resized_img.copy()
    cv2.rectangle(image_object_box, (top_bounding_box_x_min,top_bounding_box_y_max), (top_bounding_box_x_max,top_bounding_box_y_min), (255,255,255), 10)
    cv2.putText(image_object_box, top_class_label +' ' + str(top_score),
                    (top_bounding_box_x_min, top_bounding_box_y_max + 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (255, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('object detected',image_object_box)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
