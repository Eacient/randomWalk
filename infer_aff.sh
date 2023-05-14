$aff_weights=
$your_cam_dir=
$your_rw_dir=
$voc_root=

python infer_aff.py \
--weights $aff_weights \
--infer_list voc12/val.txt \
--cam_dir $your_cam_dir 
--voc12_root $voc_root
--out_rw $your_rw_dir