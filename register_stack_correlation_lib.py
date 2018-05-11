import numpy as np
import pims
from pims.bioformats import BioformatsReader
import skimage as ski
import skimage.transform as skitransform
import skimage.filters as skifilters
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
import sklearn    

class DumbInterceptEstimator:
    '''Implement a really stupid linear regression class where the slope is
    assumed to by 1, i.e. just fit the intercept. The least-squares intercept
    is simply the difference between the mean of X and the mean of y.
    In order to use RANSAC to estimate the intercept, need to provide:
    - fit(X,y) method
    - score(X,y) method
    - predict(X) method
    - get_params() method (just returns empty dict)
    - set_params() method (just raises an error to bypass that section of code)
    '''

    def fit(self,X,y):
        self.intercept_ = np.mean(y) - np.mean(X)
        return self


    def predict(self,X):
        return X + self.intercept_


    def score(self,X,y):
        y_pred = self.predict(X)
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - np.mean(y))**2).sum()
        return 1 - (ss_res / ss_tot)


    def get_params(self,deep=True):
        out = dict()
        return out


    def set_params(self,**params):
        raise ValueError


def downscale_frame(im,downscale_factor):
    '''Downscale a multidimensional image frame-by-frame. Frames are assumed to
    be the first axis in the ndarray
    '''

    if im.ndim < 3:
        raise ValueError('im must have at least 3 dimensions')
    test_downscale = skitransform.pyramid_reduce(im[0,...],
                         downscale=downscale_factor)
    downscale_shape = (im.shape[0],) + test_downscale.shape
    im_downscale = np.zeros(downscale_shape)
    for i in range(im.shape[0]):
        im_downscale[i,...] = skitransform.pyramid_reduce(im[i,...],
                                  downscale=downscale_factor)
    return im_downscale

def compute_phase_corr_stacks(im0,im1):
    '''Compute the phase correlation between two image stacks. Return a matrix
    of the maximal phase correlation between each z-slice of im0 and im1.
    '''

    max_correlation = np.zeros((im0.shape[0],im1.shape[0]))
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    for i in range(im1.shape[0]):
        fft_product = (f0 * f1[i,...].conj()) \
                      / np.abs(f0 * f1[i,...].conj())
        phase_corr_image = np.fft.ifft2(fft_product)
        phase_corr_im_flat = phase_corr_image.reshape(
                                 phase_corr_image.shape[0],-1)
        max_correlation[:,i] = np.max(phase_corr_im_flat.real,axis=1)
    return max_correlation


def fit_intercept_ransac(max_correlation):
    '''Find the intercept of the maximal correlations by RANSAC, assuming a
    slope of 1. 
    '''
    X = np.arange(max_correlation.shape[0])
    y = np.argmax(max_correlation, axis=0)
    maxima = np.max(max_correlation, axis=0)
    above_threshold = maxima >= skifilters.threshold_otsu(max_correlation)
    # Throw out peaks with low correlation
    X = X[above_threshold].reshape(-1, 1)
    y = y[above_threshold].reshape(-1, 1)
    estimator = DumbInterceptEstimator()
    ransac = linear_model.RANSACRegressor(base_estimator=estimator,
                                          residual_threshold=1.5)
    ransac.fit(X,y)
    return ransac.estimator_.intercept_


def register_frame(k, im, cumul_offset):
    '''Given the particular timepoint k and list of all cumulative z-offsets
    and the specific image im,
    return a new image that is shifted in z appropriately (and does not throw
    away any data).
    intercept is -x --> cut off first x frames and append x blank at the end
    intercept is +x --> add x blank frames in front and cut off last x frames
    '''
    min_offset = int(round(np.min(cumul_offset)))
    max_offset = int(round(np.max(cumul_offset)))
    original_len = im.shape[0]
    new_len = original_len - min_offset + max_offset
    frame_difference = int(round(cumul_offset[k]))
    im_reg_shape = (new_len,) + im.shape[1:]
    im_reg = np.zeros(im_reg_shape, dtype=im.dtype)

    new_start = frame_difference - min_offset
    new_end   = frame_difference - min_offset + original_len
    im_reg[new_start:new_end,...] = im
    return im_reg
