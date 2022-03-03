'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-02 19:50:17
Email: haimingzhang@link.cuhk.edu.cn
Description: Post process to handle the background issues
'''

import numpy as np
import face_alignment
import cv2
import subprocess
import time


def swap_masked_region(target_img, src_img, mask):
    """From src_img crop masked region to replace corresponding masked region
       in target_img
    """
    mask_img = cv2.GaussianBlur(mask, (21,21), 11)
    # input1_mask = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(mask_img))
    # input2_mask = cv2.bitwise_and(src_img, src_img, mask=mask)

    # img = cv2.addWeighted(input1_mask, 1, input2_mask, 1, 0)
    mask1 = mask_img / 255
    start = time.time()
    mask1 = np.tile(np.expand_dims(mask1, axis=2), (1,1,3))
    # print(f"mask1 time is {time.time()-start}")
    start = time.time()
    img = src_img * mask1 + target_img * (1 - mask1)
    # print(f"img time is {time.time()-start}")

    # foreground = cv2.multiply(src_img.astype(np.float), mask1)
    # background = cv2.multiply(1.0 - mask1, target_img.astype(np.float))
    # img = cv2.add(foreground, background)
    return img.astype(np.uint8)


def merge_face_contour_only(src_frame, generated_frame, face_region_coord=None):
    """Merge the face from generated_frame into src_frame
    """
    input_img = src_frame
    y1, y2, x1, x2 = 0, 0, 0, 0
    if face_region_coord is not None:
        y1, y2, x1, x2 = face_region_coord
        input_img = src_frame[y1:y2, x1:x2]

    ### 1) Detect the facial landmarks
    start = time.time()
    if not hasattr(merge_face_contour_only, 'fa'):
        merge_face_contour_only.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
        
    preds = merge_face_contour_only.fa.get_landmarks(input_img)[0] # 68x2
    # print(f"merge_face_contour_only time is {time.time()-start}")
    
    if face_region_coord is not None:
        preds += np.array([x1, y1])

    lm_pts = preds.astype(int)

    contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
    contour_pts = lm_pts[contour_idx]

    ### 2) Make the landmark region mark image
    mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)

    cv2.fillConvexPoly(mask_img, contour_pts, 255)

    ### 3) Do swap
    start = time.time()
    img = swap_masked_region(src_frame, generated_frame, mask=mask_img)
    # print(f"swap_masked_region time is {time.time()-start}")
    return img


def merge_two_face_audio(src_video, gen_video, res_video, face_region_coord=None):
    from tqdm import tqdm
    print('Start processing...')
    src_video_stream = cv2.VideoCapture(src_video)
    gen_video_stream = cv2.VideoCapture(gen_video)
    fps = src_video_stream.get(cv2.CAP_PROP_FPS)

    src_frame_count = int(src_video_stream. get(cv2.CAP_PROP_FRAME_COUNT))
    gen_frame_count = int(gen_video_stream. get(cv2.CAP_PROP_FRAME_COUNT))

    total_length = gen_frame_count
    is_gen_shorter = True
    if src_frame_count != gen_frame_count:
        total_length = min(src_frame_count, gen_frame_count)
        if src_frame_count < gen_frame_count:
            is_gen_shorter = False
        print(f"Two video length is not equal, we choose the shorter one #{total_length}")
    
    audio_from = gen_video if is_gen_shorter else src_video
    print('Extracting raw audio...')
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_from, 'temp/tmp.wav')

    subprocess.call(command, shell=True, stdout=None)

    for i in tqdm(range(total_length)):
        start = time.time()
        src_still_reading, src_frame = src_video_stream.read()
        gen_still_reading, gen_frame = gen_video_stream.read()

        if i == 0:
            frame_h, frame_w = src_frame.shape[:-1]
            print(f"Image size is {frame_h}x{frame_w}")
            out = cv2.VideoWriter('temp/tmp.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        if not src_still_reading or not gen_still_reading:
            src_video_stream.release()
            gen_video_stream.release()
            break
        start_1 = time.time()
        res_img = merge_face_contour_only(src_frame, gen_frame)
        print(f"dddd is {time.time()-start_1}")
        out.write(res_img)
        print(f"write time is {time.time()-start}")
    
    out.release()

    ### Save final synthesis face only video
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format('temp/tmp.wav', 'temp/tmp.avi', res_video)
    subprocess.call(command, shell=True, stdout=None)


if __name__ == "__main__":
    merge_two_face_audio("temp/210904_5_5.mp4", "temp/result_voice_210904_57_1.mp4", "temp/merged_face.mp4")