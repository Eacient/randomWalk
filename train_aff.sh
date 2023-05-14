$pretrained_model=
$your_crf_dir_4=
$your_crf_dir_24=
$voc_root=
$your_session_name=

python train_aff.py \
--weights $pretrained_model \
--voc12_root $voc_root \
--la_crf_dir $your_crf_dir_4 \
--ha_crf_dir $your_crf_dir_24 \
--session_name $your_session_name