set -x

generated_vertex_path="/home/haimingzhang/Research/Github/FaceFormer/work_dir/train_face_3dmm_expression_mouth_mask_same_length_data/test/lightning_logs/version_7/vis/WDA_BettyMcCollum_000_000.npy"

python main_render_pred_results.py --data_root ../data/HDTF_preprocessed \
                                   --video_name WDA_BettyMcCollum_000 \
                                   --output_root ../testing/version_1_frame_600_epoch_199_audio_WRA_LynnJenkins_000 \
                                   --gen_vertex_path ${generated_vertex_path} \
                                   --need_pose