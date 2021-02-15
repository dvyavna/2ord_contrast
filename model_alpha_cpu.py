#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import sys,time,os,glob
#import scipy.misc
from imageio import imwrite as imsave
from scipy.ndimage import zoom
import mymisc

#Размер картинки 880x880 в центре массива 1280x1280
#128, 64, 32, 16 и 8 ц/и -- число периодов первого порядка на 880
# 16,  8,  4,  2 и 1 ц/и -- число периодов второго порядка на 880
#165, 330, 660, 1320, 2640 размеры окон, включающих три периода модуляции
# 82.5 = 27.5*3
back_color=127 #140.

#Tnx 2 ali_m https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    # circular mask
    circmask = r2 <= radius*radius
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask

def ring_sector_mask(shape,centre,radius0,radius1,angle_range):
    small_mask=sector_mask(shape,centre,radius0,angle_range)
    big_mask=sector_mask(shape,centre,radius1,angle_range)
    return np.logical_xor(small_mask,big_mask)

def ring_mask(shape,centre,radius0,radius1):
    small_mask=sector_mask(shape,centre,radius0,(0,360))
    big_mask=sector_mask(shape,centre,radius1,(0,360))
    return np.logical_xor(small_mask,big_mask)

def circle_mask(shape,centre,radius):
    return sector_mask(shape,centre,radius,(0,360))

#Tnx 2 giessel https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python#7687702
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def makeGaussianRing(size, ring_mask):
    res=makeGaussian(size=size, fwhm = size)-makeGaussian(size=size, fwhm = size/3.4)
    res[ring_mask]=res[ring_mask]-np.min(res[ring_mask])
    res[ring_mask]=res[ring_mask]/np.max(res[ring_mask])
    res[~ring_mask]=0
    return res

#Tnx 2 aviraldg https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def update_progress(progress):
    print('\r[{0}] {1}%'.format('#'*(progress/10), progress))

#Tnx 2 Vladimir Ignatyev https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def zoom_in_center(arr,factor):
    #print(arr[200:200+880,200:200+880].shape)
    #Возможно, следует заменить (ndimage.)zoom на cv2.resize
    tmp_center=zoom(arr[200:200+880,200:200+880],1./factor)
    #print(tmp_center.shape)
    tmp_center_shp=np.asarray(tmp_center.shape)
    #print(tmp_center.shp+win_size*2)
    new_image=np.full(tmp_center_shp+win_size*2,mymisc.GammaDecode(back_color))
    #print(new_image.shape)
    new_image[win_size:win_size+tmp_center_shp[0],win_size:win_size+tmp_center_shp[1]]=tmp_center
    #plt.imshow(new_image, cmap='gray')
    #plt.show()
    #scipy.misc.imsave('zzzz.png', new_image)
    return new_image

def compute_maps(x,y):
    res_contrast=0.0 #res_orient=res_sf=0.0
    win=imarray[x:x+win_size,y:y+win_size]
    ave_brightness=np.mean(win)
    win = win-ave_brightness
    cwin=win*cZ
    pwin=win*pZ

    freq_cwin=np.fft.fftshift(np.abs(np.fft.fft2(cwin)))
    freq_pwin=np.fft.fftshift(np.abs(np.fft.fft2(pwin)))
    csum_contrast=np.sum(freq_cwin[freq_cmask])
    psum_contrast=np.sum(freq_pwin[freq_pmask])
    psum_sf=np.sum(freq_pwin[freq_cmask])
    res_contrast=csum_contrast-psum_contrast*CC

    #print(csum_contrast,psum_contrast)
    return res_contrast #res_orient,res_sf

#!!!!We use global vars!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
win_size=165
#24 цикла/оператор --> 8 ЦИКЛОВ НА ЦЕНТР
#12 циклов/оператор --> 4 ЦИКЛА НА ЦЕНТР?
min_lambda=6 #17
max_lambda=24 #34
CC= 0.01 #0.11 #0.14243927579653884 #0.14243891079603385 #0.025
CO= 0.01
CSF=0.01
start_time=time.time()

#Маска для вырезания частот от min_lambda до max_lambda
freq_cmask=ring_mask((win_size,win_size),(win_size//2,win_size//2),max_lambda,min_lambda)
#Маска окна для вырезания всех частот от 7 включительно (6 циклов/оператор = 2 цикла/центр)
#Сделать до 1 ц/окно?
freq_pmask=ring_mask((win_size,win_size),(win_size//2,win_size//2),1.0,win_size//2)
#Маска периферии
pmask=ring_mask((win_size,win_size),(win_size//2,win_size//2),win_size/6.0,win_size//2)
#Маска центра
cmask=circle_mask((win_size,win_size),(win_size//2,win_size//2),win_size//6.0)
#Гауссово окно для периферии
pZ=makeGaussianRing(win_size,pmask)
#Гауссово окно для центра
cZ=makeGaussian(size=win_size, fwhm = win_size//2//3)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#plt.imshow(freq_cmask, cmap='gray')
#plt.show()
#plt.imshow(freq_pmask, cmap='gray')
#plt.show()
#plt.imshow(cmask, cmap='gray')
#plt.show()
#plt.imshow(pmask, cmap='gray')
#plt.show()
#plt.imshow(cZ, cmap='gray')
#plt.show()
#plt.imshow(pZ, cmap='gray')
#plt.show()
#exit(0)

inpdir="./исходные1280x1280/" #'./input1/'
outdir='./output/'
#files=[os.path.basename(fn) for fn in glob.glob(inpdir+'*.png')]
#for fname in files:
fname=sys.argv[len(sys.argv)-1]
im = Image.open(inpdir+fname)
imarray_orig = mymisc.GammaDecode(np.array(im))
#ave_brightness=np.mean(imarray)
#imarray = imarray-ave_brightness
for scale_factor in ['01','02','04','08','16']:
    imarray=zoom_in_center(imarray_orig,int(scale_factor))
    n_x = imarray.shape[0]
    n_y = imarray.shape[1]
    #print('n_x,n_y',n_x,n_y)
    irange=n_x-win_size
    jrange=n_y-win_size
    map_contrast=np.zeros(imarray.shape) #map_orient=map_sf=np.zeros(imarray.shape)
    #print(range(irange))
    for i in range(irange):
        progress(i, irange, suffix=str(int(time.time()-start_time)))
        for j in range(jrange):
            map_contrast[i+win_size//2+1,j+win_size//2+1]=compute_maps(i,j)
            #,map_orient[i+win_size//2+1,j+win_size//2+1],map_sf[i+win_size//2+1,j+win_size//2+1]

    imsave(outdir+fname+scale_factor+'.png', map_contrast)
    #scipy.misc.imsave(outdir+fname+scale_factor+'orig.png', imarray)
    np.savez_compressed(outdir+fname+scale_factor+'.csv',map_contrast)
exit(0)

#map2=np.loadtxt('map.csv')
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, irange)
Y = np.arange(0, jrange)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, map_orient,linewidth=0, antialiased=False)
plt.show()
