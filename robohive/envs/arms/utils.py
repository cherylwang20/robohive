import os
from pathlib import Path
import time
import numpy as np
from termcolor import colored
import cv2 as cv
import matplotlib.pyplot as plt
import copy



def get_image_data(self, show=False, camera="top_down", width=200, height=200):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """

        rgb, depth = copy.deepcopy(
            self.sim.render(width=width, height=height, camera_name=camera, depth=True)
        )
        if show:
            cv.imshow("rbg", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            # cv.imshow('depth', depth)
            cv.waitKey(1)
            # cv.waitKey(delay=5000)
            # cv.destroyAllWindows()

        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

def depth_2_meters(self, depth):
    """
    Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

    Args:
        depth: The depth array to be converted.
    """

    extend = self.model.stat.extent
    near = self.model.vis.map.znear * extend
    far = self.model.vis.map.zfar * extend
    return near / (1 - depth * (1 - near / far))
