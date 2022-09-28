# product-region-detection
Built a model to automatically detect product regions in still images of shelves in supermarkets. It used a Faster R-CNN model with ResNet-18, ResNet-50 and ResNet-152 backbones respectively.
Best results were achieved with a Faster R-CNN model with ResNet-152 backbone, anchor sizes 32^2, 64^2, 128^2, 256^2, 512^2 and aspect ratios 1:2, 1:1, 2:1, which led to an average precision (AP) of 0.6863.

This project comes with a custom dataset, which comprises 100 images annotated with product regions, which can be found under pytorch_faster_rcnn/data/shelves

The paper describing the project can be found at: https://www.overleaf.com/read/zxfhybwzbtgf
Inspiration for the model was drawn from a tutorial by Johannes Schmidt: https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70
