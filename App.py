import config as cfg
import cv2
from numpy import random
from time import sleep
import playsound
import numpy as np
if cfg.WINDOWS_OS:
    import winsound
else:
    import pygame


def play_sound(sound_file):
    if cfg.WINDOWS_OS:
        winsound.PlaySound(sound_file, winsound.SND_ASYNC)
    else:
        if cfg.USE_PYGAME_AUDIO:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy() == True:
                pass
        else:
            playsound.playsound(sound_file)


def add_noise_max(img):
    # Getting the dimensions of the image
    if len(img.shape) == 3:
        row, col, _ = img.shape
    else:
        row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(500000, 999999)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def add_noise(img):
    # Getting the dimensions of the image
    if len(img.shape) == 3:
        row, col, _ = img.shape
    else:
        row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(cfg.NOISE_LIMIT_MIN, cfg.NOISE_LIMIT_MAX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(cfg.NOISE_LIMIT_MIN, cfg.NOISE_LIMIT_MAX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def add_noise2(img):
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    if len(img.shape) == 3:
        row, col, _ = img.shape
        gaus_frame = np.random.random((row, col, 3))#.astype(np.float32)
        print('gaus_frame.shape', gaus_frame.shape)
        #gaus_frame = np.expand_dims(gaus_frame, 2)
        #print(gaus_frame.shape)
        print('img.shape', img.shape)
        gauss = np.random.normal(mean, sigma, (row, col, 3))
        gauss = gauss.reshape(row, col, 3)
        im = np.empty((row, col, 3), np.uint8)
    else:
        row, col = img.shape
        gaus_frame = np.random.random((row, col))#.astype(np.float32)
        print('gaus_frame.shape', gaus_frame.shape)
        print('img.shape', img.shape)
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        im = np.empty((row, col), np.uint8)

    #frame = cv2.addWeighted(img, 0.5, gaus_frame.astype('uint8'), 0.5, 0)
    #frame = img + gauss
    frame = img + cv2.randn(im, (0), (20))

    return frame


def convert_frame(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def sound_alarm():
    print('Sounding Santa alarm...')
    play_sound(cfg.ALARM_SOUND_FILE)


def open_portal():
    print('Opening Santa portal...')
    play_sound(cfg.PORTAL_OPEN_SOUND_FILE)

    cap = cv2.VideoCapture(cfg.SANTA_PORTAL_VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_start = int(random.uniform(0, length))
    total_frames = int(cfg.PORTAL_OPEN_MAX_SECS) * 10
    msecs_between_frames = cfg.MSECS_PER_FRAME
    initial_noise_frames = cfg.INITIAL_NOISE_FRAMES

    frame_no = (random_start / (cfg.PORTAL_OPEN_MAX_SECS * fps))  # Used for jumping to frame

    grey = int(round(random.uniform(0, 1)))
    blur = int(round(random.uniform(0, 1)))
    static = int(round(random.uniform(0, 1)))

    print(86 * '=')
    print(grey, blur, static)
    print(86 * '=')
    print('FPS:', str(fps))
    print('Frames:', length)
    print('Start Frame:', random_start)
    print('Total Frames:', total_frames)
    # Read the entire file until it is completed
    frame_count_to_start = 0
    frame_count = 0

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set starting frame
    cap.set(1, random_start)
    while (cap.isOpened()):
        # Capture each frame
        ret, frame = cap.read()

        cv2.namedWindow('SantaPortal', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SantaPortal', 800, 480)

        if ret == True:
            original_frame = frame.copy()
            frame_count_to_start += 1
            if frame_count_to_start >= initial_noise_frames:
                msecs_between_frames = cfg.MSECS_PER_FRAME
                # Display the resulting frame
                if cfg.RANDOMIZE_IMAGE_ATTRIBUTES:
                    if cfg.CONVERT_IMAGE_TO_GREY and grey == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if cfg.ADD_BLUR_TO_IMAGE and blur == 1:
                        frame = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)

                    if cfg.ADD_STATIC_TO_IMAGE and static == 1:
                        frame = add_noise(frame)
                        msecs_between_frames = cfg.MSECS_PER_FRAME_NOISE
                else:
                    if cfg.CONVERT_IMAGE_TO_GREY:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if cfg.ADD_BLUR_TO_IMAGE:
                        frame = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)

                    if cfg.ADD_STATIC_TO_IMAGE:
                        frame = add_noise(frame)
                        msecs_between_frames = cfg.MSECS_PER_FRAME_NOISE

                cv2.imshow('SantaPortal', frame)
                frame_count += 1
            else:
                # Some noise
                #frame = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #frame = add_noise(frame)

                # Complete noise
                row, col = frame.shape
                frame = np.random.random((row, col, 1)).astype(np.float32)

                cv2.imshow('SantaPortal', frame)
                msecs_between_frames = 1

        if frame_count_to_start < random_start and frame_count_to_start % 10 == 0:
            print('Skipping frame:', frame_count_to_start, 'of', random_start)

        if frame_count_to_start >= random_start and frame_count % 10 == 0:
            print('Showing frame:', frame_count)

        if frame_count >= total_frames:
            frame = convert_frame(original_frame)
            cv2.imshow('SantaPortal', frame)
            print('Max frames reached, closing portal...')

        # Press Q on keyboard to exit
        if cv2.waitKey(msecs_between_frames) & 0xFF == ord('q') or not ret or frame_count >= total_frames:
            break
            # When everything done, release
            # the video capture object
            cap.release()
            # Closes all the frames
            cv2.destroyAllWindows()

    print('Closing Santa portal...')
    play_sound(cfg.PORTAL_CLOSE_SOUND_FILE)

    
def record_msg():
    print('Recording message to Santa...')


while True:
    sleeptime = random.uniform(cfg.MIN_SLEEP_SECS, cfg.MAX_SLEEP_SECS)
    print("Sleeping for:", sleeptime, "seconds")
    sleep(sleeptime)
    print("Sleeping is over...")
    sound_alarm()
    open_portal()

