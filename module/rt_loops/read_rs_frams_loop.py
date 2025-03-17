#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading

import pyrealsense2 as rs
import numpy as np


class ReadRSFrameLoop:
    """
    A simple class to continuously read aligned RealSense frames (color + depth)
    in a background thread. Frame data is stored internally and can be retrieved
    via get_latest_frames(). The frame rate is determined by the RealSense config
    (e.g., 30FPS). Additionally, depth_scale and camera intrinsic are retrieved once
    after the pipeline starts, so they can be accessed externally.
    """

    def __init__(self, freq=30):
        """
        :param freq: The desired frames per second for RealSense streaming (default 30).
                     This sets the stream config, e.g. (640x480 at 'freq' Hz).
        """
        self.freq = freq 


        self._color_image = None
        self._depth_image = None
        self._frame_lock = threading.Lock()

        self._running = True

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_scale = None  
        self.intrinsic = None  


        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.freq)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.freq)

        profile = self.pipeline.start(self.config)
        print("[RealSenseFrameLoop] RealSense pipeline started.")

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[RealSenseFrameLoop] Depth scale: {self.depth_scale}")

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                fx, fy = intr.fx, intr.fy
                cx, cy = intr.ppx, intr.ppy
                self.intrinsic = np.array([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,   0,  1]
                ], dtype=np.float32)
                print("[RealSenseFrameLoop] Intrinsic matrix:")
                print(self.intrinsic)
                break
            time.sleep(0.1)

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        while self._running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Flip horizontally (left-right) in the width dimension
            depth_image = np.fliplr(np.flipud(depth_image))
            color_image = np.fliplr(np.flipud(color_image))

            with self._frame_lock:
                self._depth_image = depth_image
                self._color_image = color_image

    def get_latest_frames(self):

        with self._frame_lock:
            return self._color_image, self._depth_image

    def get_depth_scale(self):

        return self.depth_scale

    def get_intrinsic_matrix(self):

        return self.intrinsic

    def shutdown(self):

        self._running = False
        if self._capture_thread.is_alive():
            self._capture_thread.join()

        if self.pipeline is not None:
            self.pipeline.stop()
            print("[RealSenseFrameLoop] RealSense pipeline stopped.")


if __name__ == "__main__":
    rs_loop = ReadRSFrameLoop(freq=30)

    print("[Main] depth_scale =", rs_loop.get_depth_scale())
    print("[Main] intrinsic =\n", rs_loop.get_intrinsic_matrix())

    try:
        while True:
            time.sleep(1.0)
            color_image, depth_image = rs_loop.get_latest_frames()
            if color_image is not None and depth_image is not None:
                print("[Main] Got frames:",
                      f"Color shape: {color_image.shape}, Depth shape: {depth_image.shape}")
            else:
                print("[Main] No frames yet...")
    except KeyboardInterrupt:
        rs_loop.shutdown()
        print("[Main] Shutdown complete.")
