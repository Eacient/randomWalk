$voc_root=
python evaluation.py \
--list $voc_root/ImageSets/Segmentation/val.txt \
--predict_dir $your_rw_dir \
--gt_dir $voc_root/SegmentationClassAug \
--comment $your_comments \
--type png