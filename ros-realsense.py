# import the necessary packages
# from pyimagesearch.motion_detection import SingleMotionDetector
import numpy as np
import pyrealsense2 as rs
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
colorOutputFrame = None
depthOutputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion():
    # grab global references to the video stream, output frame, and
    # lock variables
    global colorOutputFrame, depthOutputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    # md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                           cv2.COLORMAP_JET)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # grab the current timestamp and draw it on the frame
        # timestamp = datetime.datetime.now()
        # cv2.putText(color_frame, timestamp.strftime(
        #     "%A %d %B %Y %I:%M:%S%p"), (10, color_frame.shape[0] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # cv2.putText(depth_frame, timestamp.strftime(
        #     "%A %d %B %Y %I:%M:%S%p"), (10, depth_frame.shape[0] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        # if total > frameCount:
        #     # detect motion in the image
        #     motion = md.detect(gray)
        #
        #     # check to see if motion was found in the frame
        #     if motion is not None:
        #         # unpack the tuple and draw the box surrounding the
        #         # "motion area" on the output frame
        #         (thresh, (minX, minY, maxX, maxY)) = motion
        #         cv2.rectangle(frame, (minX, minY), (maxX, maxY),
        #                       (0, 0, 255), 2)
        #
        # # update the background model and increment the total number
        # # of frames read thus far
        # md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            colorOutputFrame = color_image.copy()
            depthOutputFrame = depth_colormap.copy()


def generate_color():
    # grab global references to the output frame and lock variables
    global colorOutputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if colorOutputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", colorOutputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

def generate_depth():
    # grab global references to the output frame and lock variables
    global depthOutputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if depthOutputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", depthOutputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/color_video_feed")
def color_video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_color(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/depth_video_feed")
def depth_video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_depth(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="0.0.0.0",
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=())
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
# Stop streaming
pipeline.stop()
