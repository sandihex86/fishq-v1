from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import logging
import logging.handlers


''' Logging Parameter '''
logger = logging.getLogger('fishq')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(message)s')
#handler = logging.handlers.RotatingFileHandler(filename = "fishq.log", maxBytes = 100000, backupCount= 1000)
handler = logging.handlers.TimedRotatingFileHandler(filename = 'FISHQ_DETECTED', when = 'M', backupCount = 1000 )
handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(stream_handler)


'''
FISHQ - PUSRISKAN
  Project penelitian pendeteksian objek ikan dengan menggunakan camera artifisial inteligent
  code update: 00:32 WIB 07 SEPTEMBER 2021
  UPDATE depthai 2.10.0.0
'''

# ARSITEKTUR YOLO V3
labelMap = [
    "Basket",        "Ikan",        "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "Ikan",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]
syncNN = True

# Cek direktori weights model didalam folder 
# tiny-yolo-v4_openvino_2021.2_6shave.blob
# yolov3_final_shave6.blob
nnBlobPath = str((Path(__file__).parent / Path('models/W_Pekalongan_April2021.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')
def create_pipeline():
    print("> Starting Pembuatan Pipeline FISHQ.............")
    pipeline = dai.Pipeline()

    # kamera warna
    colorCam = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    xoutRgb = pipeline.createXLinkOut()
    xoutNN = pipeline.createXLinkOut()
    xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")


    colorCam.setPreviewSize(416, 416)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # node depth
    stereo.setConfidenceThreshold(255)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.8)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)
    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(2) #set classes
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
    spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Create outputs
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    colorCam.preview.link(spatialDetectionNetwork.input)
    if(syncNN):
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        colorCam.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    print(nnBlobPath)
    return pipeline
def calculateFps(counter,startTime):
    current_time = time.monotonic()
    if (current_time - startTime) > 1 :
        fps = counter / (current_time - startTime)
        counter = 0
        startTime = current_time
    return(fps)
def visualDepth(roiDatas,depthFrame,color):
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
    for roiData in roiDatas:
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        xmin = int(topLeft.x)
        ymin = int(topLeft.y)
        xmax = int(bottomRight.x)
        ymax = int(bottomRight.y)
    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
    cv2.imshow("FISHQ-DEPTH", depthFrameColor)
def visualDetector(frame,color,detections):
    height = frame.shape[0]
    width  = frame.shape[1]
    # Denormalize bounding box
    for detection in detections:
        x1 = int(detection.xmin * width)
        x2 = int(detection.xmax * width)
        y1 = int(detection.ymin * height)
        y2 = int(detection.ymax * height)
        try:
            label = labelMap[detection.label]
        except:
            label = detection.label
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(frame, "FISHQ FPS SENSING: {:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.imshow("DEPTHAI-CAM-DETECTOR", frame)
def extractData(detections):
    for detection in detections:
        print("FISHQ-DETECTING:",str(labelMap[detection.label]),"|",
                  f"(X: {int(detection.spatialCoordinates.x)},",
                  f"Y: {int(detection.spatialCoordinates.y)},",
                  f"Y: {int(detection.spatialCoordinates.z)})mm")
def stratLogging(detections):
    for detection in detections:
        logger.info('FISHQ DETECT:{}:X={}mm:Y={}mm:Z={}mm'.format(str(labelMap[detection.label]), 
                                                                              int(detection.spatialCoordinates.x),
                                                                              int(detection.spatialCoordinates.y),
                                                                              int(detection.spatialCoordinates.x)))
class Main():
    def __init__(self):
        self.name = "FishQ apps"

    def run(self):
        pipeline = create_pipeline()
        with dai.Device(pipeline) as device:
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            color = (255, 255, 255)

            print('FISH-Q Menunggu objek yang dikenali....')
            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()
                counter+=1
                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()
                detections = inNN.detections

                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()
                    visualDepth(roiDatas,depthFrame,color)
                
                visualDetector(frame,color,detections)

                stratLogging(detections)
                
                if cv2.waitKey(1) == ord('q'):
                    break

#-------- run apps ----------

apps=Main()
apps.run()

