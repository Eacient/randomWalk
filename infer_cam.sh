$model_weights=
$your_cam_dir=
$your_crf_dir=

python infer_cam.py 
--weights $model_weights \
--infer_list voc12/val.txt\
--out_cam $your_cam_dir \
--out_crf $your_crf_dir \