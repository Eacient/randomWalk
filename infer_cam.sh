model_weights=weights/voc12_cls.pth
your_cam_dir=/roor/autodl-tmp/infer_output/vallina/train/cam
your_crf_dir=/root/autodl-tmp/infer_output/vallina/train/crf
cam_pred_dir=/root/autodl-tmp/infer_output/vallina/train/pred
voc_root=/root/autodl-tmp/VOC2012

python infer_cam.py \
--weights $model_weights \
--infer_list voc12/train.txt \
--voc12_root $voc_root \
--out_cam $your_cam_dir \
--out_crf $your_crf_dir \
--out_cam_pred $cam_pred_dir \
--out_cam_pred_alpha 0.26 \
# --sigmoid