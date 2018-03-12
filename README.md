# FasterRCNN-pytorch

FasterRCNN is implemented in VGG, ResNet and FPN base. 

reference:

rbg's FasterRCNN code: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

-----
### Before Run You Need:
1. cd ./lib 
 
   change gpu_id in make.sh and setup.py
   
   sh make.sh

    | GPU model        | Architecture    | 
    | --------   | :-----: |
    | TitanX (Maxwell/Pascal)        | sm_52      |
    | GTX 960M        | sm_50 |
    | GTX 108 (Ti)  |sm_61    |
    | Grid K520 (AWS g2.2xlarge)   |sm_30      |
    | Tesla K80 (AWS p2.xlarge)    |sm_37      |

2. cd ../
	 
   mkdir ./data
	 
   mkdir ./data/pretrained_model
	 
   download pre-trained weights in ./data/pretrained_model
3.run train.py
   
   
----
###########################

### How to use?
For example:

VGG:
CUDA_VISIBLE_DEVICES=1 python train.py --net='vgg16' --tag=vgg16 --iters=70000 --cfg='./experiments/cfgs/vgg16.yml' --weight='./data/pretrained_model/vgg16_caffe.pth'

CUDA_VISIBLE_DEVICES=2 python test.py --net='vgg16' --tag=vgg16 --model=60000 --cfg='./experiments/cfgs/vgg16.yml' --model_path='voc_2007_trainval/vgg16/vgg16_faster_rcnn' --imdb='voc_2007_test' --comp

ResNet:

CUDA_VISIBLE_DEVICES=2 python train.py --net='res18' --tag=res18 --iters=70000 --cfg='./experiments/cfgs/res18.yml' --weight='./data/pretrained_model/Resnet18_imagenet.pth'

CUDA_VISIBLE_DEVICES=3 python train.py --net='res50' --tag=res50 --iters=70000 --cfg='./experiments/cfgs/res50.yml' --weight='./data/pretrained_model/Resnet50_imagenet.pth'

CUDA_VISIBLE_DEVICES=7 python train.py --net='res101' --tag=res101 --iters=80000 --cfg='./experiments/cfgs/res101.yml' --weight='./data/pretrained_model/resnet101_caffe.pth'

CUDA_VISIBLE_DEVICES=6 python test.py --net='res101' --tag=res101_1 --cfg='./experiments/cfgs/res101.yml' --model=70000 --model_path='voc_2007_trainval/res101_1' --imdb='voc_2007_test' --comp

----
