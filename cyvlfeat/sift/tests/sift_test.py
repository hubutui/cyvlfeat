from __future__ import division
import cyvlfeat
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.sift.sift import sift
from cyvlfeat.sift.phow import phow
import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.test_util import lena
from skimage.io import load_sift, imread
from skimage.util import img_as_float32
import os.path as osp
from scipy.spatial.distance import cdist


img = lena().astype(np.float32)
half_img = img[:, :256]


# helper function just like vl_ubcread
def ubc_read():
    sift_path = osp.join(osp.dirname(cyvlfeat.__file__), 'data', 'box.sift')
    data = load_sift(sift_path)
    ubc_descriptors = data['data']
    ubc_frames = np.column_stack([data['row'], data['column'], data['scale'], data['orientation']])

    return ubc_frames.astype(np.float32), ubc_descriptors.astype(np.float32)

def test_ubc_descriptor():
    img_path = osp.join(osp.dirname(cyvlfeat.__file__), 'data', 'box.pgm')
    img = imread(img_path)
    img = img.astype(np.float32)
    ubc_frames, ubc_descriptors = ubc_read()
    [frames, descriptors] = sift(img, first_octave=-1, frames=ubc_frames, compute_descriptor=True, float_descriptors=True)
    distance = cdist(frames, ubc_frames)
    # tranpose these array so that I could easily convert MATLAB code to numpy
    distance = distance.transpose()
    frames = frames.transpose()
    descriptors = descriptors.transpose()
    ubc_frames = ubc_frames.transpose()
    ubc_descriptors = ubc_descriptors.transpose()
    
    perm = distance.argmin(axis=0)
    frames = frames[:, perm]
    descriptors = descriptors[:, perm]
    t1 = np.mean(np.sqrt(np.sum((ubc_descriptors - descriptors)**2, axis=0)))
    t2 = np.mean(np.sqrt(np.sum(ubc_descriptors**2, axis=0)))
    err = t1 / t2
    assert err < 0.1


def test_phow_float_descriptors():
    i = img.copy()
    frames, descriptors = phow(i, float_descriptors=True)
    assert descriptors.dtype == np.float32


def test_phow_frames():
    i = img.copy()
    frames, descriptors = phow(i, float_descriptors=True)
    assert frames.shape[0] == 236206


def test_phow_descriptors():
    i = img.copy()
    frames, descriptors = phow(i, float_descriptors=True)
    assert descriptors.shape[0] == 236206


def test_dsift_non_float_descriptors():
    i = img.copy()
    frames, descriptors = dsift(i, float_descriptors=False)
    assert descriptors.dtype == np.uint8


def test_dsift_float_descriptors():
    i = img.copy()
    frames, descriptors = dsift(i, float_descriptors=True)
    assert descriptors.dtype == np.float32


def test_dsift_steps():
    i = half_img.copy()
    # Step 3 in Y-Direction, 4 in X-Direction
    frames, descriptors = dsift(i, step=[3, 4])

    assert frames.shape[0] == 10416
    assert_allclose(frames[:3], [[4.5, 4.5], [4.5, 8.5], [4.5, 12.5]],
                    rtol=1e-3)


def test_dsift_windowsize():
    i = half_img.copy()
    frames, descriptors = dsift(i, window_size=3)

    assert frames.shape[0] == 124241
    assert_allclose(frames[:3], [[4.5, 4.5], [4.5, 5.5], [4.5, 6.5]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, -3:], [74, 55, 71],
                    rtol=1e-3)


def test_dsift_fast():
    i = half_img.copy()
    frames, descriptors = dsift(i, fast=True)

    assert frames.shape[0] == 124241
    assert_allclose(frames[:3], [[4.5, 4.5], [4.5, 5.5], [4.5, 6.5]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, -3:], [61, 45, 60],
                    rtol=1e-3)


def test_dsift_norm():
    i = half_img.copy()
    frames, descriptors = dsift(i, norm=True)

    assert frames.shape[-1] == 3
    assert frames.shape[0] == 124241
    assert_allclose(frames[:3], [[4.5, 4.5, 1.6537], [4.5, 5.5, 1.7556],
                                 [4.5, 6.5, 1.8581]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, -3:], [65, 48, 62],
                    rtol=1e-3)


def test_sift_n_frames():
    i = img.copy()
    frames = sift(i)
    assert_allclose(frames[0], [2.16217, 128.056, 2.13029, 5.9325], rtol=1e-3)
    assert frames.shape[0] == 730


def test_sift_non_float_descriptors():
    i = half_img.copy()
    frames, descriptors = sift(i, compute_descriptor=True)

    assert_allclose(frames[0], [2.16217, 128.056, 2.13029, 5.9325], rtol=1e-3)
    assert_allclose(descriptors[0, 84:88], [4,  14,  32, 137])
    assert frames.shape[0] == 358


def test_sift_user_defined_frames():
    i = half_img.copy()
    frames = np.array([[4., 5., 2., 0], [3., 2., 3., np.pi / 2],
                       [8., 6., 4., 1.]])
    new_frames, descriptors = sift(i, frames=frames, compute_descriptor=True)

    assert_allclose(new_frames[0], frames[0], rtol=1e-3)
    assert_allclose(descriptors[0, -5:], [14, 22, 23, 10,  2])
    assert new_frames.shape[0] == 3


def test_sift_sort_user_defined_scales():
    i = half_img.copy()
    frames = np.array([[4., 5., 3., 0], [3., 2., 2., np.pi / 2],
                       [8., 6., 1., 1.]])
    new_frames, descriptors = sift(i, frames=frames, compute_descriptor=True)

    assert_allclose(new_frames[0], frames[-1], rtol=1e-3)
    assert_allclose(descriptors[0, -5:], [46, 14,  0,  0,  0])
    assert new_frames.shape[0] == 3


def test_sift_force_orientations():
    i = half_img.copy()
    frames = np.array([[4., 5., 2., 0], [3., 2., 3., np.pi / 2],
                       [8., 6., 4., 1.]])
    new_frames, descriptors = sift(i, frames=frames, compute_descriptor=True,
                                   force_orientations=True)

    assert_allclose(new_frames[0], [4, 5, 2, 4.6239], rtol=1e-3)
    assert_allclose(descriptors[0, :5], [5, 1, 0, 0, 1])
    assert new_frames.shape[0] == 3
