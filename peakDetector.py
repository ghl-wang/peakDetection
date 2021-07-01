# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:36:17 2020

@author: WenboWang
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:36:21 2020

@author: WenboWang

This is a transcribed version of feature extractors in Matlab

"""

'''
This function extracts features for regresssion analysis

function args:
    
    stripImg: numpy array, rgb image after narrowing and binning
    channel: integer, channel number

'''
import cv2
import numpy as np
from scipy.signal import savgol_filter, detrend,find_peaks
import matplotlib.pyplot as plt


''' 
calculate the area under the curve
'''
def calculate_AUC(curve, peak_location):
    curve_length=len(curve)
    
    loc_lower=peak_location-int(0.117*curve_length)
    loc_upper=peak_location+int(0.091*curve_length)+1
    
    if loc_lower>0 and loc_upper<curve_length:
        peak_section = curve[loc_lower:loc_upper]
    elif loc_lower<=0:
        peak_section = curve[:loc_upper]
    else:
        peak_section = curve[loc_lower:]
        
    AUC = np.trapz(peak_section)
    return AUC
    
'''

''' 

def calculate_noise(curve,cloc):
    start=cloc+50
    endloc=cloc+150
    curve_section=curve[start:endloc]
    curve_section_detrended=detrend(curve_section)
    stdCurve=np.std(curve_section_detrended)
    return stdCurve
    
    
def calculate_peak_baseline(curve,peak_location):
    curve_length=len(curve)
    
    cutoff_lower = int(0.117*curve_length)
    cutoff_upper = int(0.091*curve_length)
    
    loc_lower=peak_location-cutoff_lower
    loc_upper=peak_location+cutoff_upper+1
    
    xgap=loc_upper-loc_lower
    
    if loc_lower>0 and loc_upper<curve_length:
        ygap = curve[loc_upper]-curve[loc_lower];
        baseline=(cutoff_lower/xgap)*ygap+curve[loc_lower];
    elif loc_lower<=0:
        ygap=curve[loc_upper]-curve[0];
        xgap_lower=peak_location;
        baseline=(xgap_lower/(xgap_lower+cutoff_upper))*ygap+curve[0];
    else:
        ygap=curve[-1]-curve[loc_lower];
        xgap_upper=curve_length-peak_location;
        baseline=(xgap_upper/(xgap_upper+cutoff_lower))*ygap++curve[loc_lower];
    
    PeakminusBase=curve[peak_location]-baseline;
    
    return PeakminusBase

def plotPeaks(curve,locs,peaks,title,savefolder):
    plt.figure()    
    plt.plot(curve)
    plt.scatter(locs,peaks,c='r',marker='d') 
    plt.title(title)
    filename=title.split('\n')[0]
    savepath=r'C:\Users\WenboWang\SpyderProjects\LODpeakDetect'+'/'+savefolder+'/'+filename
    plt.savefig(savepath)
    # plt.show()



def extractor3(stripImg, channel=0,win_size=11,file_tag='No tag',savefolder='figs'):
    
    # depending on the value of channel, take either single channel data or
    # convert the image to grayscale image
    # 0,1,2: BGR channels
    # -2: YUV's intensity channel
    # -3: HSL's Luminescence channel
    # other values: grayscale image
    
    if -1<channel<3:
        Img_for_1dcurve=stripImg[:,:,channel]
    elif channel==-2:
        img_YUV=cv2.cvtColor(stripImg,cv2.COLOR_BGR2YUV)
        Img_for_1dcurve=img_YUV[:,:,0]
    elif channel==-3:
        img_HSV=cv2.cvtColor(stripImg,cv2.COLOR_BGR2HSV)
        Img_for_1dcurve=img_HSV[:,:,2]
    elif channel==-4:
        img_HLS=cv2.cvtColor(stripImg,cv2.COLOR_BGR2HLS)
        Img_for_1dcurve=img_HLS[:,:,1]
    elif channel==-5: #RpGoB
        Img_for_1dcurve= (400*((stripImg[:,:,1].astype(np.float32)+stripImg[:,:,2].astype(np.float32))/stripImg[:,:,0].astype(np.float32)-1.4)).astype(np.uint8)        
    else:
        Img_for_1dcurve = cv2.cvtColor(stripImg, cv2.COLOR_BGR2GRAY)
    
    strip_curve  =  Img_for_1dcurve.mean(1)
    
    # # use this line to test unsmoothed version
    # strip_curve_smoothed = strip_curve  
    
    strip_curve_smoothed  =  savgol_filter(strip_curve, win_size, 2) # window size=11, order=3   
    strip_curve_smoothed_inverted =  255 - strip_curve_smoothed  
    strip_curve_smoothed_inverted_detrend = \
        detrend(strip_curve_smoothed_inverted)
    
    curve_length = len(strip_curve_smoothed_inverted_detrend)
    half_curve_length = round(curve_length/2)
    
    # calculate pseudo-absorbance
    basevalue_at_half = \
        strip_curve_smoothed[half_curve_length-3:half_curve_length+3].mean()
    strip_Reflectance = strip_curve_smoothed/basevalue_at_half
    strip_Absorbance = 1/strip_Reflectance
    strip_Absorbance_log = np.log10(strip_Absorbance)
    strip_Absorbance_exp = 10**strip_Absorbance
    
    # base Value to subtract from inverted and detrended curve, to 'zero' the
    # baseline of inverted curve.
    
    basevalue_at_half_afterDetrend = \
        strip_curve_smoothed_inverted_detrend[half_curve_length-3:half_curve_length+3].mean()
    strip_curve_inverted_detrended_baselinezeroed = \
        strip_curve_smoothed_inverted_detrend - basevalue_at_half_afterDetrend
    
    peak_distance=int(40/90*curve_length)
    peak_width=(3/90)*curve_length
    
    locs,props=\
        find_peaks(strip_curve_inverted_detrended_baselinezeroed,height=1.0,distance=peak_distance,width=peak_width)
        
        
    if len(locs)>0 and locs[0]<half_curve_length:
        floor_noise=calculate_noise(strip_curve_smoothed_inverted_detrend,locs[0])
    
    if len(locs) == 0:
        result_curve = [strip_curve, strip_curve_inverted_detrended_baselinezeroed,np.array([-1,-1])]
        result_array = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        
        # plot non-detected curve
        plot_title=file_tag+'\n'+'None detected'
        plotPeaks(strip_curve_inverted_detrended_baselinezeroed, locs, props['peak_heights'],plot_title,savefolder)

        
        return result_curve+result_array
    
    # control line location
    control_line_location = locs[0]
    
    # change if condition parameters to relative scales
    if len(locs)==2:
        peak_distance_relative=(locs[1]-locs[0])/len(strip_curve)
    else:
        peak_distance_relative=0.6 # send the code for one peak or non-peak if statements
        
    
    # when two dominant peaks were found, there is no need to calculate peak height
    if len(locs) == 2 and peak_distance_relative<0.65:
        signal_line_location = locs[1]
        
        control_peak_height = props['peak_heights'][0]
        signal_peak_height = props['peak_heights'][1]
        
        control_peak_AUC=\
            calculate_AUC(strip_curve_inverted_detrended_baselinezeroed, control_line_location)
        signal_peak_AUC=\
            calculate_AUC(strip_curve_inverted_detrended_baselinezeroed, signal_line_location)
        
        control_peak_height_linear_baseline = \
            calculate_peak_baseline(strip_curve_smoothed_inverted, control_line_location)
        signal_peak_height_linear_baseline = \
            calculate_peak_baseline(strip_curve_smoothed_inverted, signal_line_location)
        
        control_peak_absorbance_normalized = \
            calculate_peak_baseline(strip_Absorbance, control_line_location)
        signal_peak_absorbance_normalized = \
            calculate_peak_baseline(strip_Absorbance, signal_line_location)
        
        # line 70 in featureExtractor3.m
        signal_peak_absorbance_log_normalized = \
            calculate_peak_baseline(strip_Absorbance_log, signal_line_location)
        signal_peak_absorbance_exp_normalized = \
            calculate_peak_baseline(strip_Absorbance_exp, signal_line_location)
            
        control_peak_height_raw = strip_curve_smoothed_inverted[control_line_location]
        signal_peak_height_raw = strip_curve_smoothed_inverted[signal_line_location]
        
        snr_value=signal_peak_height_linear_baseline/floor_noise
        plot_title='Both control and test lines detected!'
    # when only one peak is found and it is the control line peak    
    elif control_line_location<half_curve_length:
        signal_line_offset=int(0.468*curve_length)
        
        signal_line_location = -1
        
        control_peak_height = props['peak_heights'][0]
        signal_peak_height = -1
  
        
        control_peak_AUC=\
            calculate_AUC(strip_curve_inverted_detrended_baselinezeroed, control_line_location)
        signal_peak_AUC=0
        
        control_peak_height_linear_baseline = \
            calculate_peak_baseline(strip_curve_smoothed_inverted, control_line_location)
        signal_peak_height_linear_baseline = -1
        
        control_peak_absorbance_normalized = \
            calculate_peak_baseline(strip_Absorbance, control_line_location)
        signal_peak_absorbance_normalized = \
            calculate_peak_baseline(strip_Absorbance, signal_line_location)
        
        # line 70 in featureExtractor3.m
        signal_peak_absorbance_log_normalized = \
            calculate_peak_baseline(strip_Absorbance_log, signal_line_location)
        signal_peak_absorbance_exp_normalized = \
            calculate_peak_baseline(strip_Absorbance_exp, signal_line_location)
            
        control_peak_height_raw = strip_curve_smoothed_inverted[control_line_location]
        signal_peak_height_raw = strip_curve_smoothed_inverted[signal_line_location]       
        snr_value=-0.01
        plot_title='Only control line detected!'
    # only one peak detected and it is assumed to be signal line    
    elif control_line_location/curve_length<0.81:
        control_line_location = 0
        signal_line_location = locs[0]
        
        control_peak_height = 0.01
        signal_peak_height = props['peak_heights'][0]
        
        control_peak_AUC=0.01
        signal_peak_AUC=\
            calculate_AUC(strip_curve_inverted_detrended_baselinezeroed, signal_line_location)
        
        control_peak_height_linear_baseline = 0.01
        signal_peak_height_linear_baseline = \
            calculate_peak_baseline(strip_curve_smoothed_inverted, signal_line_location)
        
        control_peak_absorbance_normalized = 0.01
        signal_peak_absorbance_normalized = \
            calculate_peak_baseline(strip_Absorbance, signal_line_location)
        
        # line 70 in featureExtractor3.m
        signal_peak_absorbance_log_normalized = \
            calculate_peak_baseline(strip_Absorbance_log, signal_line_location)
        signal_peak_absorbance_exp_normalized = \
            calculate_peak_baseline(strip_Absorbance_exp, signal_line_location)
            
        control_peak_height_raw = 0.01
        signal_peak_height_raw = strip_curve_smoothed_inverted[signal_line_location]
        snr_value=0.01
        
        plot_title='Only test line detected'
        
        # invalid entry
    else:
        
        control_line_location = 0
        signal_line_location = 0
        
        control_peak_height = 0.01
        signal_peak_height = 0.01
        
        control_peak_AUC=0.01
        signal_peak_AUC=0.01
        
        control_peak_height_linear_baseline = 0.01
        signal_peak_height_linear_baseline = 0.01
        
        control_peak_absorbance_normalized = 0.01
        signal_peak_absorbance_normalized = 0.01
        
        # line 70 in featureExtractor3.m
        signal_peak_absorbance_log_normalized = 0.01
        signal_peak_absorbance_exp_normalized = 0.01
            
        control_peak_height_raw = strip_curve_smoothed_inverted[0]
        signal_peak_height_raw = strip_curve_smoothed_inverted[-1]
        snr_value=0.01
        plot_title='Invalid!'
    
    # plot the curve
    plotPeaks(strip_curve_inverted_detrended_baselinezeroed, locs, props['peak_heights'],file_tag+'\n'+plot_title,savefolder)

    
    result_curve = [strip_curve, strip_curve_inverted_detrended_baselinezeroed,\
                    np.array([control_line_location,signal_line_location])]
    
    result_array = [control_peak_height,signal_peak_height,\
                    control_peak_AUC,signal_peak_AUC,\
                    control_peak_height_linear_baseline,signal_peak_height_linear_baseline,\
                        control_peak_absorbance_normalized,signal_peak_absorbance_normalized,\
                            signal_peak_absorbance_log_normalized,\
                                signal_peak_absorbance_exp_normalized,\
                                    control_peak_height_raw,signal_peak_height_raw,snr_value]

        
    return result_curve+result_array

        
