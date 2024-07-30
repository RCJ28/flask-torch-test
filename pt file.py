print('Program started')
import torch
import cv2

def load_model(ckpt_path, conf=0.10, iou=0.10):
    model = torch.hub.load('.','custom',
                           path_or_model=ckpt_path,source='local',force_reload=True)
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    #model.classes = [0,1,2,3,4]
     # maximum number of detections per image
    return model

def draw_boxes(r,img,classes):
    x,y,width,height=0,0,0,0
    label_class=0
    count = 0
    for i in range(len(r.pred[0].data)):
        x,y,width,height = r.pred[0].data[i][:4].detach().cpu().numpy()
        count += 1
        label_class = int(r.pred[0].data[i][5])
        #print("label",label_class)
        #if label_class == 0:
        #cv2.rectangle(img, (int(x),int(y-20)), (int(width),int(y)), (255, 144, 30), -1)
        #cv2.putText(img, classes[label_class], (int(x),int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)
        cv2.rectangle(img,(int(x),int(y)),(int(width),int(height)),(255,0,255),thickness=2)
        
        # cv2.imshow("frame",img)
        # cv2.waitKey(0)
    return img,count


model_path = "./pipe_count_model.pt"
model = load_model(model_path)


image = cv2.imread('./input_image.jpg')
results = model(image)

img,count = draw_boxes(results,image,model.names)
