echo "Installing requirements..."
pip install -r requirements.txt
echo "Downloading Tiny ImageNet..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping Tiny ImageNet..."
unzip -qq tiny-imagenet-200.zip
cp -R tiny-imagenet-200 tiny-images
echo "Divide Validation folder to Validation / test (50/50) ..."
mv tiny-imagenet-200/test tiny-imagenet-200/test_save
python3 prepare_data.py
python3 prepare_no_split.py
mkdir models
mkdir out
echo "Copying all images to ./tiny-224..."
cp -r tiny-imagenet-200 tiny-224
echo "Resizing images to 224 x 224..."
python3 resize.py

cp tiny-images/words.txt ./words.txt


