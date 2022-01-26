echo "Installing requirements..."
pip install -r requirements.txt
echo "Downloading Tiny ImageNet..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping Tiny ImageNet..."
unzip -qq tiny-imagenet-200.zip
echo "Divide Validation folder to Validation / test (50/50) ..."
python3 prepare_data.py
mkdir models
echo "Copying all images to ./tiny-224..."
cp -r tiny-imagenet-200 tiny-224
echo "Resizing images to 224 x 224..."
python3 resize.py

