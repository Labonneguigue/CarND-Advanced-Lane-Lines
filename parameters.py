'''
Set of parameters
Keeping them here facilitates tunning my processing pipeline
'''
parameters = { 'orig_points_x' : (575, 705, 1127, 203),#(617, 660, 1125, 188),
               'orig_points_y' : 460, #434,
               's_thresh'      : (170, 255),
               'sx_thresh'     : (20, 100),
               'l_white_thresh': (200, 255),
               'margin'        : 100,
               'detection_distance' : 0,
               'videofile_in'  : 'project_video.mp4',
               'videofile_out' : 'output_videos/project_video.mp4',
            #    'videofile_in'  : 'challenge_video.mp4',
            #    'videofile_out' : 'output_videos/challenge_video.mp4',
               'output_video_as_images' : 0,
               'debug_output'  : 1
}

# src = np.float32(
#     [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
#     [((img_size[0] / 6) - 10), img_size[1]],
#     [(img_size[0] * 5 / 6) + 60, img_size[1]],
#     [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
# dst = np.float32(
#     [[(img_size[0] / 4), 0],
#     [(img_size[0] / 4), img_size[1]],
#     [(img_size[0] * 3 / 4), img_size[1]],
#     [(img_size[0] * 3 / 4), 0]])
