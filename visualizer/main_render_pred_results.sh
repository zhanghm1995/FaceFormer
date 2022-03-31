set -x

generated_vertex_path="/home/haimingzhang/Research/Github/FaceFormer/work_dir/train_face_3dmm_expression_mouth_mask/lightning_logs/version_9/vis/WDA_MartinHeinrich_000_000.npy"

python main_render_pred_results.py --data_root ../data/HDTF_preprocessed \
                                   --video_name WDA_MartinHeinrich_000 \
                                   --output_root ../testing/debug2 \
                                   --gen_vertex_path ${generated_vertex_path} \
                                   --need_pose