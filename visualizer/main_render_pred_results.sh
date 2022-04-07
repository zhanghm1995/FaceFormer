set -x

generated_vertex_path="/home/haimingzhang/Research/Github/FaceFormer/work_dir/train_face_3dmm_expression_mouth_mask_paper/test/lightning_logs/version_12/vis/train/WDA_ChrisVanHollen1_000_000.npy"

python main_render_pred_results.py --data_root ../data/HDTF_face3dmmformer/val \
                                   --video_name WDA_ChrisVanHollen1_000 \
                                   --output_root ../testing/paper_val_WDA_ChrisVanHollen1_000_000 \
                                   --gen_vertex_path ${generated_vertex_path} \
                                   --need_pose


##### Old version ###########
# generated_vertex_path="/home/haimingzhang/Research/Github/FaceFormer/work_dir/train_face_3dmm_expression_mouth_mask_paper/test/lightning_logs/version_1/vis/train/WDA_BarackObama_000_000.npy"

# python main_render_pred_results.py --data_root ../data/HDTF_preprocessed \
#                                    --video_name WDA_BarackObama_000_000 \
#                                    --output_root ../testing/paper_val_WDA_BarackObama_000_000 \
#                                    --gen_vertex_path ${generated_vertex_path} \
#                                    --need_pose