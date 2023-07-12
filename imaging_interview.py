import cv2
import imutils

import os

# given
def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img

# given
def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray

# given
def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh

# my code for filtering unique images
def filter_unique_images(
        data_path='/Users/sid/Downloads/kopernikus/dataset',
        min_cnt_area = 0.5
    ):
    # list all files in the folder
    files = [
        name for name in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, name)) 
            and not name.startswith('.DS')
    ]

    # dictionary with filename and its shape
    files_shapes = {
        x: img.shape for x in files
        if (img := cv2.imread(os.path.join(data_path, x))) is not None
    }

    files = files_shapes.keys()

    cameras = set([x[:3] for x in files])
    # maintain dictionaries for each camera
    camera_files = {
        camera : [x for x in files if x[:3] == camera]
        for camera in cameras
    }
    camera_shapes = {
        key: [
            files_shapes[x] for x in value
        ]
        for key, value in camera_files.items()
    }
    camera_promiment_shape = {
        key : max(set(value), key=value.count)
        for key, value in camera_shapes.items()
    }
    # the algorithm provided needs images to be of same shape for comparison
    # and so, remove the minority of images of different shape 
    camera_files = {
        key: sorted([
            x for x in value
            if files_shapes[x] != camera_promiment_shape[key]
        ])
        for key, value in camera_files.items()
    }
    # two pointer approach for comparing if images are similar or not
    # using principle of locality of reference as the idea behind this approach (as dataset is sorted according to timesteps)
    for key, value in camera_files.items():
        num_files = len(value)
        if num_files == 0:
            print(f'{key}: {value}')
        prev = value[0]
        prev_img = cv2.imread(os.path.join(data_path, prev))
        prev_preprocessed = preprocess_image_change_detection(prev_img)
        curr = value[0]
        accepted_files = []
        accepted_files.append(prev)
        for i in range(1, num_files):
            curr = value[i]
            curr_img = cv2.imread(os.path.join(data_path, curr))
            curr_preprocessed = preprocess_image_change_detection(curr_img)
            score, _, _ = compare_frames_change_detection(prev_preprocessed, curr_preprocessed, min_cnt_area)
            if score > 80000:
                accepted_files.append(curr)
                prev = curr
                prev_img = curr_img
                prev_preprocessed = curr_preprocessed
        camera_files[key] = accepted_files
    return camera_files

# this will return a list of unique images needed for the object detection algorithm
filter_unique_images()

