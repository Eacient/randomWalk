$model_weights=
$your_cam_dir=
$your_crf_dir=
$cam_pred_dir=

python infer_cam.py 
--weights $model_weights \
--infer_list voc12/val.txt\
--out_cam $your_cam_dir \
--out_crf $your_crf_dir \
--out_cam_pred $cam_pred_dir \
--out_cam_pred_alpha 0.26 \
--sigmoid