# MaskR-CNN_Road-Markings
  ## 1. Label the objects
  Using VGG Image Annotator (VIA) annotaions technique
  ## 2. Mask R-CNN environment
  * Operating System: Ubuntu 18.04
  * GPU: NVIDIA GeForce RTX2080 Ti
  * CUDA 9.1
  * Tensorflow-gpu 1.12
  * python 3.6
  ## 3. Modify config.py `bev_project.py`
  * Modify number of classes (classes number + background)
  * Modify batch size: `IMAGES_PER_GPU`*`GPU_COUNT`
  * Modify the iterations per epoch: `STEPS_PER_EPOCH` (according to the training samples// batch size)
  * Modify the threshold of the confidence scores:`DETECTION_MIN_CONFIDENCE`
  * Modify the classes name `add_class`
  
  ## 4. Training
      python3 bev_project.py train --weights=coco --dataset=/home/geo/kh/Mask_RCNN-master_sign/samples/sign_data
  
  ## 5. Evaluation `bev_mAP.py`
  * The latest weight path: `MODEL_DIR` `COCO_MODEL_PATH`
  * Modify the `class_names`  
      python3 bev_mAP.py
      
  ## 6. Visulization `bev_test_project.py`
  * The save path of images:`save_dir`
  * The latest weight path: `MODEL_DIR` `COCO_MODEL_PATH`
  * Modify the `class_names`
  
      python3 bev_test_project.py
      
  ## 7. Confusion Matrix `Confusion_Matrix_bev.py`
  * The save path of images:`save_dir`
  * The latest weight path: `MODEL_DIR` `COCO_MODEL_PATH`
  * Modify the `class_names`
      python3 Confusion_Matrix_bev.py
