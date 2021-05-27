#!/bin/bash

# Automatically installs pyspark
#
#
# 1. Pull this file
# 2. run: chmod +x spark_install.sh
# 3. run: sudo bash ./spark_install.sh

# install pip3
echo ''
echo '----------------Installing pip3------------------'
echo ''
apt-get -q install python3-pip -y

# # Install jdk/jre-14
echo ''
echo '----------------Installing jdk/jre-----------------'
echo 'Please wait...'
echo 'Installing jdk. Please wait...'
echo ''
apt install -qq openjdk-14-jdk-headless -y



# # Get spark version: 3.1.1 (spark-3.1.1-bin-hadoop2.7)
echo ''
echo '----------------Downloading spark------------------'
echo 'Please wait...'
wget -q https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop2.7.tgz

# # untar the files
echo ''
echo '--------------Extracting Spark Folder--------------'
echo ''
tar -xzf spark-3.1*
rm -rf spark-3.1.1-bin-hadoop2.7.tgz

# make a dir for spark
echo ''
echo '--------Making spark dir @ /usr/local/spark--------'
echo ''
mkdir /usr/local/spark

# move it to /usr/local/spark
echo ''
echo '--------Moving spark dir to /usr/local/spark--------'
echo ''
mv spark-3.1* /usr/local/spark

# install pyspark with pip3
echo ''
echo '----------Installing pyspark------------'
echo ''
pip3 install pyspark

# configuring paths
echo 'SPARK_HOME="/usr/local/spark/spark-3.1.1-bin-hadoop2.7'>> .bashrc
echo 'PATH=${PATH}:$SPARK_HOME/bin'>> .bashrc
echo 'PYSPARK_PYTHON="/usr/bin/python3"'>> .bashrc
echo 'PYSPARK_DRIVER_PYTHON="/usr/bin/python3"'>> .bashrc
source .bashrc
echo 'Done.'

