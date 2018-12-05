######################################
# INSTALL OPENCV ON UBUNTU OR DEBIAN #
######################################

# |          THIS SCRIPT IS TESTED CORRECTLY ON          |
# |------------------------------------------------------|
# | OS               | OpenCV       | Test | Last test   |
# |------------------|--------------|------|-------------|
# | Ubuntu 18.04 LTS | OpenCV 3.4.2 | OK   | 18 Jul 2018 |



# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove

# install utilities 
sudo apt-get install tmux -y
sudo apt-get install htop -y

# 2. INSTALL THE DEPENDENCIES

# Build tools:
#sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
#sudo apt-get install -y qt5-default libvtk6-dev



# INSTALL THE git
sudo apt install git-all -y


# INSTALL pip
#sudo apt-get install python-setuptools python-dev build-essential -y

#sudo easy_install pip

# python scientific 
# Interpreter and the package manager:
#sudo apt-get install -y python3 python3-dev python3-pip

# Essential scientific libraries:
#sudo apt-get install -y python3-numpy python3-matplotlib python3-scipy python3-pandas python3-simpy

# IPython:
#sudo apt-get install -y ipython3 ipython3-notebook



# INSTALL python for tensorflow
sudo apt -y install python-dev python-pip

# cudnn for cuda 9.0  ubuntu
wget -O cudnn-9.0-linux-x64-v7.3.0.29.tar https://www.dropbox.com/s/ezlfz6xccpydkbu/cudnn-9.0-linux-x64-v7.3.0.29.tar?dl=0

tar -xvf cudnn-9.0-linux-x64-v7.3.0.29.tar

sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/

sudo cp -P cuda/include/* /usr/local/cuda/include/

sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
#check CuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# INSTALL python for tensorflow

pip install --user --upgrade tf-nightly-gpu  

# test tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
# test gpu device
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"


#### model part
# github
git clone https://github.com/qoo/facenet.git

# install requirement
pip install -r requirement.txt

# Set the python path
export PYTHONPATH=~/facenet/src

# run test 
python ~/facenet/triplet_loss_test.py

# Download the LFW dataset

mkdir ~/datasets
mkdir -p ~/datasets/lfw/raw
tar xvf ~/lfw.tar -C ~/datasets/lfw/raw --strip-components=1



# Align the LFW dataset 
for N in {1..4}; do \
python ~/facenet/src/align/align_dataset_mtcnn.py \
~/datasets/lfw/raw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done

# Download pre-trained model (optional)
mkdir -p ~/models/facenet
tar xvf ~/2018model.tar -C ~/models/facenet 

# Run the test

python ~/facenet/src/validate_on_lfw.py \
~/datasets/lfw/raw \
~/models/facenet/20180408-102900 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization


python ~/facenet/src/validate_on_lfw.py \
~/datasets/lfw/raw \
~/models/facenet/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization


# Prepare dataset
mkdir -p ~/datasets/casia/CASIA-maxpy-clean
tar xvf ~/CASIA-WebFace.tar -C ~/datasets/casia/CASIA-maxpy-clean --strip-components=1

ls ~/datasets/casia/CASIA-maxpy-clean/._*
rm ~/datasets/casia/CASIA-maxpy-clean/._*


# Start training
python ~/facenet/src/train_tripletloss.py \
--logs_base_dir ~/logs/facenet/ \
--models_base_dir ~/models/facenet/ \
--data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160 \
--optimizer RMSPROP \
--learning_rate 0.01 \
--weight_decay 1e-4 \
--max_nrof_epochs 500

