#!/usr/bin/env bash

#wget -O [saved_file_name] [direct_download_link]

#GoTurn
wget -O ./weights/goturn.pth.tar https://umich.box.com/shared/static/src6rfm4lpn0v3t4l26d6u0v4ixdwem5.tar

#SSD
wget -O ./weights/ssd300_mAP_77.43_v2.pkl https://umich.box.com/shared/static/jszcnnwcvscfyqe3o81xy8qzfbsc20vo.pkl

#C3D
wget -O ./weights/c3d-pretrained.pth https://umich.box.com/shared/static/znmyt8uph3w7bjxoevg7pukfatyu3z6k.pth

#C3D Mean
wget -O ./weights/sport1m_train16_128_mean.npy https://umich.box.com/shared/static/ppbnldsa5rty615osdjh2yi8fqcx0a3b.npy 
