##########################################################################

# Example : DoG saliency demo - [Katramados / Breckon 2011]

# This implementation:
# Copyright (c) 2020 Ryan Lail, Toby Breckon, Durham University, UK

##########################################################################

import cv2
import argparse
import sys
import math

##########################################################################

from saliencyDoG import SaliencyDoG

##########################################################################


if __name__ == "__main__":

    keep_processing = True
    toggle_saliency = True
    toggle_time_info = True
    frame_timestamp = 0

    # parse command line arguments for camera ID or video file

    parser = argparse.ArgumentParser(
        description='Perform ' +
        sys.argv[0] +
        ' example operation on incoming camera/video image')
    parser.add_argument(
        "-c",
        "--camera_to_use",
        type=int,
        help="specify camera to use",
        default=0)
    parser.add_argument(
        "-r",
        "--rescale",
        type=float,
        help="rescale video by this factor",
        default=1.0)
    parser.add_argument(
        "-fs",
        "--fullscreen",
        action='store_true',
        help="run in full screen mode")
    parser.add_argument(
        "-g",
        "--grayscale",
        action='store_true',
        help="process frames as grayscale")
    parser.add_argument(
        "-l",
        "--low_pass_filter",
        action='store_true',
        help="apply a low_pass_filter to saliency map")
    parser.add_argument(
        "-m",
        "--multi_layer_map",
        action='store_true',
        help="use every layer in the production of the saliency map")
    parser.add_argument(
        'video_file',
        metavar='video_file',
        type=str,
        nargs='?',
        help='specify optional video file')
    args = parser.parse_args()

    ##########################################################################

    # define video capture object

    try:
        # to use a non-buffered camera stream (via a separate thread)

        if not (args.video_file):
            import camera_stream
            cap = camera_stream.CameraVideoStream()
        else:
            cap = cv2.VideoCapture()  # not needed for video files

    except BaseException:
        # if not then just use OpenCV default

        print("INFO: camera_stream class not found - camera input may be "
              "buffered")
        cap = cv2.VideoCapture()

    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG(ch_3=not (args.grayscale),
                                  low_pass_filter=args.low_pass_filter,
                                  multi_layer_map=args.multi_layer_map)

    # define display window name

    window_name = "Live Input"  # window name

    # if command line arguments are provided try to read video_name
    # otherwise default to capture from attached camera

    if (((args.video_file) and (cap.open(str(args.video_file))))
            or (cap.open(args.camera_to_use))):

        # create window by name (as resizable)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while (keep_processing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount()

            # if camera /video file successfully open then read frame

            if (cap.isOpened):
                ret, frame = cap.read()
                timestamp_latest = cap.get(cv2.CAP_PROP_POS_MSEC)

                # check the timestamp of the frame (and skip if not new)

                if (timestamp_latest == frame_timestamp):
                    continue  # skip identical frames
                else:
                    cap_fps = 1000 / (timestamp_latest - frame_timestamp)
                    frame_timestamp = timestamp_latest

                # when we reach the end of the video (file) exit cleanly

                if (ret == 0):
                    keep_processing = False
                    continue

                # rescale if specified

                if (args.rescale != 1.0):
                    frame = cv2.resize(
                        frame, (0, 0), fx=args.rescale, fy=args.rescale)

            # perform saliency processing via Division of Gaussians
            # [Katramados / Breckon 2011]

            if toggle_saliency:
                frame = saliency_mapper.generate_saliency(frame)

            # stop the timer and convert to ms. (to see how long processing and
            # display takes)

            stop_t = ((cv2.getTickCount() - start_t) /
                      cv2.getTickFrequency()) * 1000

            if toggle_time_info:
                label = ('Processing time: %.0f ms' % stop_t) + \
                        (' [ Max. framerate (processing): %.0f fps' %
                         (1000 / stop_t)) + ' ]'
                cv2.putText(frame, label, (0, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
                label = ('Supplied framerate (camera): %.0f fps' % cap_fps)
                cv2.putText(frame, label, (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
            # display image

            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN & args.fullscreen)

            # start the event loop + wait 40ms or less depending on
            # processing time taken (i.e. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

            # detect specific key strokes by recording which key is pressed

            # - "x" - exit
            # - "f" - fullscreen
            # - "s" - toggle saliency display on/off
            # - "t" - toggle fps/time info display

            if (key == ord('x')):
                keep_processing = False
            elif (key == ord('f')):
                args.fullscreen = not (args.fullscreen)
            elif (key == ord('s')):
                toggle_saliency = not (toggle_saliency)
                toggle_time_info = not (toggle_time_info)
            elif (key == ord('t')):
                toggle_time_info = not (toggle_time_info)

        # close all windows

        cv2.destroyAllWindows()

    else:
        print("No video file specified or camera connected.")

##########################################################################
