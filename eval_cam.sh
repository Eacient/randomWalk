voc_root=/root/autodl-tmp/VOC2012
your_cam_dir=output/vallina/cam
your_comments=vallina-val-cam

python evaluation.py \
--list voc12/val_name.txt \
--predict_dir $your_cam_dir \
--gt_dir $voc_root/SegmentationClassAug \
--comment $your_comments \
--type npy \
--curve True
