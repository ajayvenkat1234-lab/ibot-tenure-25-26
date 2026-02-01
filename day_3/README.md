V.AJAY

SCOPE OF THIS PROJECT:
To create a classification model which can distinguish cats and dogs from the images fed in.

FINAL TEST ACCURACY : 99%

DATA AUGMENTATION TECHNIQUES USED:
    1) RandomCrop
    2) RandomHorizontalFlip
    3) RandomRotation
    4) RandomAffine
    5) ColorJitter

Learning rates schedule used: ReduceLROnPlateau for initial training and constant learning rates for fine tuning

Challenges faced:
1) Understanding the pipeline and the terminologies used in its
2) Initially i was getting all cats for the correct and wrong prediction examples so had to change it to 5 cats and 5 dogs for a  more robust analysis
3) Increasing accuracy of MobileNetV2 

Bonus Challenges:
1) Fine tuning: Unfreezing of layer4 in ResNet and fixing learning rate of layer4 and classification layer to 0.0001 and 0.001 respectively

2) Try different Architecture: Accuracy of ResNet was 99% and that of MobileNetV2 was 96.80% but after unfreezing the last 2 layers and classification layer in MobileNetV2 an accuracy of 98.56% was achieved

This happens because the last layer in ResNet has more parameters than the last layer in MobileNetV2 so effectively we will be tuning more parameters and this leads to increased accuracy 
Each layer in MobileNetV2 is lightweight and contains less parameters compared to ResNet 

3) Visualize Predictions: Displayed a Grid of Correct predictions(5 cats and 5 dogs) anf Incorrect predictions(5 cats and 5 dogs)
