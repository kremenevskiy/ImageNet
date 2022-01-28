Result of correct classes on validation on Summary.ipunb

Model pretrained on Kaggle Notebooks on: https://www.kaggle.com/kremenevskiy/tiny-imagenet/notebook

with Accuracy on validation: 0.68


When run
python inference.py imgs_path file.csv
imgs_path must contain at least one subfolder.
Example: images stored in tiny-images/val/images 
then run with:
python inference.py tiny-images/val file.csv


By default we apply transforms.Resize(224) wich is so expensive for CPU. 
Advice: apply resize to (224 x 224) for all images first.
Then run model with out transforms.Resize(224).


Difficult part of the project was that the amount of data provided was too small.
Due to limit on amount of data, the final test accuracy could not mark above 68%. 
Furthur study on image augmentation may help improving the
performance of the model.


For that project was used ResNet18. And it was tested with sampe hyperparameteres. 
We can move to another Network, for example to Efficient Net which can give more than 80% accuracy on validation.

Using different Network, optimizer, loss function, and augmentation method may improve the final result.


