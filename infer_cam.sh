model_weights=weights/voc12_cls.pth
your_cam_dir=output/vallina/cam
your_crf_dir=output/vallina/crf
cam_pred_dir=output/vallina/pred
voc_root=/root/autodl-tmp/VOC2012

python infer_cam.py \
--weights $model_weights \
--infer_list voc12/val.txt \
--voc12_root $voc_root \
--out_cam $your_cam_dir \
--out_crf $your_crf_dir \
--out_cam_pred $cam_pred_dir \
--out_cam_pred_alpha 0.26 \
# --sigmoid