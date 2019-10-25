#!/usr/bin/env bash

#wget -O [saved_file_name] [direct_download_link]

#SSD
wget -O ./weights/ssd300_mAP_77.43_v2.pkl https://umich.box.com/shared/static/jszcnnwcvscfyqe3o81xy8qzfbsc20vo.pkl

#C3D
wget -O ./weights/c3d-pretrained.pth https://umich.box.com/shared/static/znmyt8uph3w7bjxoevg7pukfatyu3z6k.pth

#C3D Mean
wget -O ./weights/sport1m_train16_128_mean.npy https://umich.box.com/shared/static/ppbnldsa5rty615osdjh2yi8fqcx0a3b.npy 

#YC2BB-Full model
wget -O ./weights/yc2bb_full-model.pth https://umich.box.com/shared/static/5ukbdcawryzkkq4r789z0src6u6uvg3u.pth 

#I3D pretrained on ImageNet and then Kinetics by original authors
wget -O ./weights/i3d_rgb_imagenet.pt https://umich.box.com/shared/static/5m6dwwepzdcw3kjhx7s0peb59lbcde0s.pt

#I3D pretrained on ImageNet, Kinetics, then on HMDB51 in ViP
wget -O ./weights/i3d_rgb_imagenet_then_HMDB51_30epochs.pkl https://umich.box.com/shared/static/x8x83sw4htidxsxgtus9nt00f383mmm7.pkl
