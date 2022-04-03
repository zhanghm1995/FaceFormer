set -x

checkpoint="/home/haimingzhang/Research/Github/FaceFormer/work_dir/train_face_3dmm_expression_mouth_mask_same_length_data/lightning_logs/version_1/checkpoints/epoch=199-step=107599.ckpt"
python main_train_one_hot.py --cfg config/face_3dmm_expression_mouth_mask_test.yaml --test_mode \
                             --checkpoint ${checkpoint} \
                             --checkpoint_dir work_dir/train_face_3dmm_expression_mouth_mask_same_length_data/test