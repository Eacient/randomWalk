$voc_root
python evaluation.py \
--list VOC2012/ImageSets/Segmentation/val.txt \
--predict_dir $your_cam_dir \
--gt_dir $voc_root/SegmentationClassAug \
--comment $your_comments \
--type npy \
--curve True
