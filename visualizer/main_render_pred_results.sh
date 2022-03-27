set -x

python main_render_pred_results.py --data_root ../data/HDTF_preprocessed \
                                   --video_name WDA_MartinHeinrich_000 \
                                   --gen_vertex_path /home/zhanghm/Research/Github/FaceFormer/work_dir/train_face_3dmm_vertex_mouth_mask/test/lightning_logs/version_1/vis/WDA_MartinHeinrich_000_000.npy