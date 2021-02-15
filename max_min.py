#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import exposure
from scipy import spatial
from photutils import find_peaks
import cv2
import psychopy.visual
import psychopy.filters
#filters.butter2d_bp
import mymisc
from skimage.draw import line,line_aa
# https://stackoverflow.com/questions/55013954/how-to-apply-a-gabor-filter-to-an-image-with-hexagonal-sampling
# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, phi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio =0.5
# phi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold
# added: bandwidth=2.

def GetGaborSigma(g_lambda,bandwidth):
    return g_lambda * (1./np.pi*np.sqrt(np.log(2.) / 2.) * ((2.**bandwidth +1.) / (2.**bandwidth-1.)))

def GetGaborFuncArrayDim(gamma,sigma):
    #Расчёт размеров фильтра по N. Petkov and M.B. Wieling, Groningen University
    #compute the size of the 2n+1 x 2n+1 matrix to be filled with the values of a Gabor function
    #this size depends on sigma and gamma
    if (gamma <= 1. and gamma >= 0.):
        return np.ceil(2.5*sigma/gamma)*2+1
    else:
        return np.ceil(2.5*sigma)*2+1

def AdjustGaborKernel(kernel):
    pos_sum= np.sum(kernel[kernel>=0.])
    neg_sum=np.abs(np.sum(kernel[kernel<=0.]))
    mean_sum=(pos_sum+neg_sum)/2.0
    if (mean_sum >= 0.0):
        pos_sum=pos_sum/mean_sum
        neg_sum=neg_sum/mean_sum
    kernel[kernel>=0.]*=neg_sum
    kernel[kernel<=0.]*=pos_sum
    return kernel

def GaborFilterUsing6t(img,la):
    sigma=GetGaborSigma(la,bandwidth)
    g_size=int(GetGaborFuncArrayDim(gamma,sigma))
    gabor_coef=(1./(2.*sigma*sigma))/np.pi
    filtering_result = np.zeros_like(img)
    #print 'np.min(img)',np.min(img),'np.max(img)',np.max(img)
    for th in thetas:
        for pshift in pshifts:
            g_kernel = cv2.getGaborKernel((g_size, g_size), sigma, th, la, gamma, pshift, ktype=cv2.CV_32F)
            #plt.imshow(np.fft.fftshift(np.abs(np.fft.fft2(g_kernel))), cmap='gray')
            #plt.imshow(g_kernel, cmap='gray')
            #plt.show()
            g_kernel=AdjustGaborKernel(g_kernel)*gabor_coef
            #print 'np.sum(g_kernel)',np.sum(g_kernel),'np.min(g_kernel)',np.min(g_kernel),'np.max(g_kernel)',np.max(g_kernel)
            filtered_img = cv2.filter2D(img, cv2.CV_32F, g_kernel)
            #print 'np.mean(filtered_img)',np.mean(filtered_img),'np.min(filtered_img)',np.min(filtered_img),'np.max(filtered_img)',np.max(filtered_img)
            #plt.imshow(filtered_img, cmap='gray')
            #plt.show()
            #Объединение sum:
            filtering_result+=filtered_img
            #Объединение max:
            #filtering_result=np.where(abs(filtering_result)<abs(filtered_img),filtered_img,filtering_result)
    #filtering_result/=len(lambdas)*len(pshifts)*4. #(4.0 -- эмпирическое значение)
    #plt.imshow(filtering_result, cmap='gray')
    #print 'np.mean(filtering_result)',np.mean(filtering_result),'np.min(filtering_result)',np.min(filtering_result),'np.max(filtering_result)',np.max(filtering_result)
    #plt.show()
    return filtering_result

def RemoveNearestPoints(lst,r):
    tree=spatial.KDTree(lst)
    q=tree.query_pairs(r)
    #q=sorted(q)
    #print 'q',q
    #tmp_crd=q[:][1]
    c = map(lambda x: x[1], q)
    tmp_crd=list(set(list(c)))
    #print 'tmp_crd',tmp_crd
    lst=np.delete(lst,tmp_crd,axis=0)
    #print 'shape of lst', lst.shape
    tmp_crd=list()
    tmp_layer=np.zeros([transp_mask.shape[0]+booble_offset*2,transp_mask.shape[1]+booble_offset*2])
    tmp_layer=tmp_layer.astype('bool')
    #print np.sum(tmp_layer)
    tmp_layer[booble_offset:booble_offset+transp_mask.shape[0],booble_offset:booble_offset+transp_mask.shape[1]]=np.logical_not(transp_mask)
    #plt.imshow(tmp_layer, cmap='gray')
    #plt.show()
    for n in range(lst.shape[0]):
        if np.sum( np.logical_and( booble_mask,tmp_layer[np.int(lst[n,0]):np.int(lst[n,0])+booble_extent,
            np.int(lst[n,1]):np.int(lst[n,1])+booble_extent]) ) == 0:
            tmp_crd.append(n)
    #print tmp_crd
    lst=np.delete(lst,tmp_crd,axis=0)
    return lst

def getDist(x1,y1,x2,y2):
    import math
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

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

def rebuild_vector(vect):
    len1=len(vect)//2
    arr1,arr2 = np.split(vect,[len1])
    arr1=np.flip(arr1)
    return np.insert(arr2,range(len(arr1)),arr1)

def find_points(fname,l):
    #obj_size=(110.0*spa_ratio)/2.0
    obj_size=(220.0*spa_ratio)/2.0

    curr_obj_size=(l*spa_ratio)/2.0
    #win_count=int(obj_size**2/curr_obj_size**2)
    win_count=int(obj_size/curr_obj_size)
    #win_count=win_counts[l]
    #print 'obj_size',obj_size
    #print 'curr_obj_size',curr_obj_size
    #print 'win_count',win_count

    mapZ=np.load(fname)['arr_0']
    #print 'mapZ.shape', mapZ.shape
    scaled_mapZ=np.zeros([center_x+win_size*2,center_y+win_size*2])
    #print 'scaled_mapZ.shape', scaled_mapZ.shape
    imsize=mapZ.shape
    #tmp_center=cv2.resize(mapZ[win_size/2+1:imsize[1]-win_size/2,win_size/2+1:imsize[0]-win_size/2],
    #    (center_x+win_size,center_y+win_size),interpolation=cv2.INTER_LANCZOS4)
    tmp_center=cv2.resize(mapZ[win_size+1:imsize[1]-win_size,win_size+1:imsize[0]-win_size],
        (center_x,center_y),interpolation=cv2.INTER_LANCZOS4)
    #print 'tmp_center.shape',tmp_center.shape
    #print 'mapZ[win_size/2+1:imsize[1]-win_size/2,win_size/2+1:imsize[0]-win_size/2].shape', mapZ[win_size/2+1:imsize[1]-win_size/2,win_size/2+1:imsize[0]-win_size/2].shape
    #print 'tmp_center.shape', tmp_center.shape
    #scaled_mapZ[win_size/2+1:win_size/2+1+880+win_size,win_size/2+1:win_size/2+1+880+win_size]=tmp_center
    scaled_mapZ[win_size+1:win_size+1+880,win_size+1:win_size+1+880]=tmp_center
    #plt.imshow(scaled_mapZ, cmap='gray')
    #plt.show()
    mapZ=scaled_mapZ
    #Иначе шум:
    #mapZ/=10.
    mapZ=np.round(mapZ,0)

    #min_max_mask=transp_mask #(mapZ==0)
    min_max_mask=(mapZ==0)
    #np.savetxt('tmp_mask.csv',min_max_mask)
    #np.savetxt('tmp.csv',mapZ)
    mapZ = (exposure.rescale_intensity(mapZ,out_range=(0.0,1.0)))#+1.)
    #np.savetxt('tmp.csv',mapZ)
    #print 'np.min(mapZ)',np.min(mapZ)
    #print 'np.max(mapZ)',np.max(mapZ)

    thresholds=0.0 #np.full(mapZ.shape,0.0)
    #print 'mapZ.shape',mapZ.shape
    #coordinates_max=find_peaks(mapZ,thresholds,box_size=box_sizes[l],border_width=win_size,mask=min_max_mask)
    #coordinates_max=find_peaks(mapZ,thresholds,box_size=l,border_width=win_size,mask=min_max_mask)
    #coordinates_max=find_peaks(mapZ,thresholds,box_size=l,border_width=win_size,mask=min_max_mask,npeaks=1024)
    coordinates_max=find_peaks(mapZ,thresholds,border_width=win_size,mask=min_max_mask,npeaks=1024)
    #Сортировка по возрастанию, преобразование в массив:
    coordinates_max.sort(keys='peak_value')
    #Печатается таблица!
    #print(coordinates_max)
    #Next two lines are for Python2 or 3
    #coordinates_max=np.asarray(map(list, zip(*[coordinates_max['y_peak'],coordinates_max['x_peak'],coordinates_max['peak_value']])))
    coordinates_max=np.asarray(list(zip(*[coordinates_max['y_peak'],coordinates_max['x_peak'],coordinates_max['peak_value']])))
    #Меняем порядок сортировки
    #print('crd_max',coordinates_max)
    coordinates_max=np.flip(coordinates_max,0)

    mapZ_inv=mapZ*-1+1#+2
    #np.savez_compressed('tmp1.csv',mapZ_inv)
    #print 'np.min(mapZ_inv)',np.min(mapZ_inv)
    #print 'np.max(mapZ_inv)',np.max(mapZ_inv)

    #coordinates_min=find_peaks(mapZ_inv,thresholds,box_size=box_sizes[l],border_width=win_size,mask=min_max_mask)
    #print ('box_sizes[l]',box_sizes[l])
    #coordinates_min=find_peaks(mapZ_inv,thresholds,box_size=l,border_width=win_size,mask=min_max_mask)
    #coordinates_min=find_peaks(mapZ_inv,thresholds,box_size=l,border_width=win_size,mask=min_max_mask,npeaks=1024)
    coordinates_min=find_peaks(mapZ_inv,thresholds,border_width=win_size,mask=min_max_mask,npeaks=1024)
    try:
        coordinates_min.sort(keys='peak_value')
        coordinates_min=np.asarray(map(list, zip(*[coordinates_min['y_peak'],coordinates_min['x_peak'],coordinates_min['peak_value']])))
        coordinates_min=np.flip(coordinates_min,0)
        #print coordinates_min
        #plt.imshow(mapZ, cmap='gray')

    except:
        coordinates_min=np.asarray([[0,0,0.]])
        #plt.imshow(mapZ, cmap='gray')
        #plt.imshow(min_max_mask, cmap='gray')

    #print 'len(coordinates_max)', len(coordinates_max)
    #print 'len(coordinates_min)', len(coordinates_min)
    #plt.imshow(mapZ, cmap='gray')
    #plt.imshow(min_max_mask, cmap='gray')

    #Удалим близко лежащие точки
    #print(len(coordinates_max))
    coordinates_max=RemoveNearestPoints(coordinates_max, r=gauss_radius*0.84/2.8) #r=5.)
    #print(len(coordinates_max))
    coordinates_min=RemoveNearestPoints(coordinates_min, r=gauss_radius*0.84/2.8) #r=5.)
    #print 'gauss_radius',gauss_radius
    #q=sorted(q)
    #tmp_crd=sorted(list(set(list(c))),reverse=True)

    #print 'len(coordinates_max) после прореживания', len(coordinates_max)
    #print 'len(coordinates_min) после прореживания', len(coordinates_min)

    #Можно сделать разные количества разных пиков
    #max_peaks_count=np.min([win_count,len(coordinates_max)])
    #min_peaks_count=np.min([win_count,len(coordinates_min)])
    #print 'max_peaks_count',max_peaks_count
    #print 'min_peaks_count',min_peaks_count
    #Но мы сделаем одинаково, чтобы площадь открытого изображения не зависела от их числа и т.д.
    extrema_count=np.min([win_count,len(coordinates_max),len(coordinates_min)])
    #print 'extrema_count',extrema_count

    #!!!!!!!!!!!!!!
    #coordinates_max=np.asarray([[600.,200.,1.]])
    #coordinates_min=np.asarray([[1000.,800.,1.]])
    #extrema_count=1

    #Упорядочиваем максимумы и минимумы
    #print map(list,zip(*[coordinates_min[:,0],coordinates_min[:,1]]))
    #tmpcrd=zip(*[coordinates_min[:extrema_count,0],coordinates_min[:extrema_count,1]])
    tmp_crd_min=zip(*[coordinates_min[:extrema_count,0],coordinates_min[:extrema_count,1]])
    new_coordinates_min=list()
    new_coordinates_max=zip(*[coordinates_max[:extrema_count,0],coordinates_max[:extrema_count,1]])
    #print new_coordinates_max[0]
    #exit(0)
    for i in range(extrema_count):
        tree=spatial.KDTree(tmp_crd_min)
        q=tree.query(new_coordinates_max[i])
        new_coordinates_min.append(tmp_crd_min[q[1]])
        tmp_crd_min.pop(q[1])
    #print new_coordinates_min
    #print zip(*[coordinates_min[:extrema_count,0],coordinates_min[:extrema_count,1]])
    #print new_coordinates_max

    #Соединительные линии
    #plt.plot([x1,x2],[y1,y2],'k-')
#    for i in range(extrema_count):
#        plt.plot([new_coordinates_max[i][1],new_coordinates_min[i][1]],
#            [new_coordinates_max[i][0],new_coordinates_min[i][0]],'g-')#'k-')

    #Ищем точки, лежащие между ближайшими максимумами и минимумами и имеющие амплитуду, близкую к средней этих двух
    new_coordinates_ave=list()
    coordinates_ave=np.zeros_like(coordinates_max)
    for i in range(extrema_count):
        ave_ampl=np.mean([mapZ[int(new_coordinates_max[i][0]),int(new_coordinates_max[i][1])],
            mapZ[int(new_coordinates_min[i][0]),int(new_coordinates_min[i][1])]])
        ampl_diff=mapZ[int(new_coordinates_max[i][0]),int(new_coordinates_max[i][1])]-mapZ[int(new_coordinates_min[i][0]),int(new_coordinates_min[i][1])]
        rr, cc, br_val = line_aa(int(new_coordinates_max[i][0]), int(new_coordinates_max[i][1]),
            int(new_coordinates_min[i][0]), int(new_coordinates_min[i][1]))
        #plt.scatter(cc,rr)
        #print 'mapZ[int(tmp_xcoord),int(tmp_ycoord)]',mapZ[int(tmp_xcoord),int(tmp_ycoord)]
        #print 'mapZ[new_coordinates_max[i][0],new_coordinates_max[i][1]]', mapZ[int(new_coordinates_max[i][0]),int(new_coordinates_max[i][1])]
        #print 'bright_diff',bright_diff
        #plt.scatter(cc,rr,color='g')
        initial_opt_val=getDist(1279.,1279.,0.,0.)*1.
        #print 'getDist(1279.,1279.,0.,0.)*1.',getDist(1279.,1279.,0.,0.)*1.
        new_coordinates_ave.append([0,0])
        for j in rebuild_vector(range(len(rr))): #range(len(rr)):
            dist_to_max=getDist(new_coordinates_max[i][0],new_coordinates_max[i][1],rr[j],cc[j])
            dist_to_min=getDist(new_coordinates_min[i][0],new_coordinates_min[i][1],rr[j],cc[j])
            ave_dist=np.abs(dist_to_max-dist_to_min)
            #print ave_dist
            #bright_diff=np.abs(ave_ampl-mapZ[rr[j],cc[j]])
            curr_bright_percent=(mapZ[rr[j],cc[j]]
                -mapZ[int(new_coordinates_min[i][0]),int(new_coordinates_min[i][1])])/ampl_diff
            #print curr_bright_percent
            if 0.4<curr_bright_percent<0.6:
                #if ave_dist*abs(curr_bright_percent-0.5)<initial_opt_val:
                initial_opt_val=ave_dist*abs(curr_bright_percent-0.5)
                coordinates_ave[i,0]=rr[j]
                coordinates_ave[i,1]=cc[j]
                coordinates_ave[i,2]=mapZ[rr[j],cc[j]]
                #new_coordinates_ave[len(new_coordinates_ave)-1]=[rr[j],cc[j],mapZ[rr[j],cc[j]]]
                new_coordinates_ave[len(new_coordinates_ave)-1]=[rr[j],cc[j],curr_bright_percent]
                break
        coordinates_ave[i,2]=(coordinates_ave[i,2]-mapZ[int(new_coordinates_min[i][0]),int(new_coordinates_min[i][1])])/ampl_diff
        #print 'coordinates_ave[i,2]',coordinates_ave[i,2]
        #plt.plot([rr[j],cc[j]], marker='o', color='r', ls='')

    #plt.plot(coordinates_max[:max_peaks_count, 1], coordinates_max[:max_peaks_count, 0], 'r.')
    #plt.plot(coordinates_min[:min_peaks_count, 1], coordinates_min[:min_peaks_count, 0], 'g.')

    #plt.plot(coordinates_max[:extrema_count, 1], coordinates_max[:extrema_count, 0], 'r.')
    #plt.plot(coordinates_ave[:extrema_count, 1], coordinates_ave[:extrema_count, 0], 'g.')
    #plt.plot(coordinates_min[:extrema_count, 1], coordinates_min[:extrema_count, 0], 'b.')
    #plt.show()
    #return coordinates_max,coordinates_ave,coordinates_min
    #print 'min[0,0]',mapZ[int(new_coordinates_min[0][0]),int(new_coordinates_min[0][1])]
    #print 'max[0,0]',mapZ[int(new_coordinates_max[0][0]),int(new_coordinates_max[0][1])]
    return new_coordinates_max,new_coordinates_ave,new_coordinates_min

def ButterworthFilter(img,la):
    #128, 64, 32, 16 и 8 ц/и -- число периодов первого порядка на 880
    #Максимальная частота для изображения 880x880 составляет 440/изобр.
    #При размере 1280x1280 шаг изменения частоты составляет 0,6875 для 880x880
    cfreq=880.0/la #делим размер изображения на длину волны, получаем ц/изобр.
    #print cfreq/440.
    #print 'cfreq',cfreq

    #low_cut =mymisc.incr_oct(cfreq,-1)/440.*0.5
    #high_cut=mymisc.incr_oct(cfreq,1)/440.*0.5
    low_cut =mymisc.incr_oct(cfreq,-0.5)/440.*0.5
    high_cut=mymisc.incr_oct(cfreq,0.5)/440.*0.5
    #print 'lower',low_cut,'upper',high_cut
    filt = psychopy.visual.filters.butter2d_bp(size=img.shape,cutin=low_cut,cutoff=high_cut,n=10)
    #plt.imshow(filt, cmap='gray')
    #plt.show()
    img_freq = np.fft.fft2(img)
    img_filtered = np.fft.fftshift(img_freq) * filt
    img_new  = np.real(np.fft.ifft2(np.fft.ifftshift(img_filtered)))
    #plt.imshow(img_new, cmap='gray')
    #plt.show()
    #print 'np.min(img_new)',np.min(img_new),'np.max(img_new)',np.max(img_new),'np.sum(img_new)',np.sum(img_new)
    return img_new

#x=np.array([[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]])

#if len(sys.argv)<2:
#    print 'Argument required!'
#    exit(0)
#print sys.argv[1]

center_x=880
center_y=880
win_size=165
spa_ratio=8 #16

#lambdas=[6.875,13.75,27.5,55.0,110.0]
lambdas=[13.75,27.5,55.0,110.0,220.0]
win_counts={13.75:128,27.5:32,55.0:8,110.0:2,220.0:1}
box_sizes={13.75:3,27.5:6,55.0:12,110.0:24,220.0:48}
thetas=np.arange(0, np.pi, np.pi / 6.)
pshifts=np.asarray([0.,np.pi/2.])
bandwidth=2.0
gamma=0.5

inpdir='./исходные1280x1280/'
outdir='./output/'
#img_fn='a_f_n_00130_931230_fb.png'
img_fn=sys.argv[len(sys.argv)-1]
img = mymisc.GammaDecode(cv2.imread(inpdir+img_fn,cv2.IMREAD_GRAYSCALE).astype(float))
#img = cv2.imread(inpdir+img_fn,cv2.IMREAD_GRAYSCALE).astype(float)

transp_mask = cv2.imread('./masks/'+img_fn,cv2.IMREAD_GRAYSCALE).astype(float)
#Изображение -- 255, фон -- 0
transp_mask=np.array(transp_mask,dtype=bool)
#Предполагается, что center_x==center_y и (1280-center_x-2*win_size) -- чётное число
mask_border_offset=(1280-center_x-2*win_size)//2
#print mask_border_offset
transp_mask=np.logical_not(transp_mask[mask_border_offset:1280-mask_border_offset,mask_border_offset:1280-mask_border_offset])
#print('np.min(transp_mask),np.max(transp_mask)',np.min(transp_mask),np.max(transp_mask))
#print(transp_mask[640,640])
#print(transp_mask[0,0])
#print transp_mask.shape
#exit(0)

ave_brightness=np.mean(127.)
img-=ave_brightness
#print 'np.mean(img)',np.mean(img)

files=[outdir+img_fn+'01.csv.npz',outdir+img_fn+'02.csv.npz',
    outdir+img_fn+'04.csv.npz',outdir+img_fn+'08.csv.npz',outdir+img_fn+'16.csv.npz']

#print 'np.min(img)',np.min(img),'np.max(img)',np.max(img)

#for la in lambdas:
#    #GaborFilterUsing6t(img,la)
#    #ButterworthFilter(img,la)
#    pass

#exit(0)

max_images=dict()
min_images=dict()
ave_images=dict()

#for a in reversed(zip(*[files,lambdas])):
cntr=0
for a in zip(*[files,lambdas]):
    cntr+=1
    #print a
    gauss_radius=a[1]/2.0/2.0*spa_ratio
    #print 'gauss_radius',gauss_radius
    booble=makeGaussian(gauss_radius*2, fwhm = gauss_radius*.84)
    #plt.imshow(booble, cmap='gray')
    #plt.show()

    #Круги!
    ##booble=np.where((0.02<booble) & (booble<0.05),1.,0.)
    booble_mask=np.where(booble>0.02,1.,0.)
    #booble=np.where(0.04<booble,1.,0.)

    booble_shape=booble.shape
    booble_extent=booble_shape[0]
    booble_offset=booble_extent//2
    #print 'booble.shape',booble_shape
    #plt.imshow(booble, cmap='gray')
    #plt.show()
    coord_max,coord_ave,coord_min=find_points(a[0],a[1])
    #print(coord_max)
    offset=(1280-1210)/2
    #image_layer=GaborFilterUsing6t(img,a[1])
    #image_layer=img
    image_layer=ButterworthFilter(img,a[1])
    #print 'np.max(image_layer),np.min(image_layer)', np.max(image_layer),np.min(image_layer)
    gauss_layer_max=np.zeros([img.shape[0]+booble_offset*2,img.shape[1]+booble_offset*2])
    gauss_layer_min=np.zeros([img.shape[0]+booble_offset*2,img.shape[1]+booble_offset*2])
    gauss_layer_ave=np.zeros([img.shape[0]+booble_offset*2,img.shape[1]+booble_offset*2])
    #print 'gauss_layer_max.shape',gauss_layer_max.shape
    for i in reversed(range(len(coord_max))):
        #Нужно помнить, что координаты смещаются за счёт добавления 35 пикселей с каждой стороны картинки (+offset),
        #половины ширины гауссиана с каждой стороны картинки (+booble_offset),
        #а также за счёт вычитания половины ширины гауссиана из координат с целью центровки (-booble_offset).
        gauss_layer_max[int(coord_max[i][0]+offset):int(coord_max[i][0]+offset)+booble_shape[0],
            int(coord_max[i][1]+offset):int(coord_max[i][1]+offset)+booble_shape[1]]=np.where(gauss_layer_max[int(coord_max[i][0]+offset):int(coord_max[i][0]+offset)+booble_shape[0],
            int(coord_max[i][1]+offset):int(coord_max[i][1]+offset)+booble_shape[1]]>booble,
            gauss_layer_max[int(coord_max[i][0]+offset):int(coord_max[i][0]+offset)+booble_shape[0],
            int(coord_max[i][1]+offset):int(coord_max[i][1]+offset)+booble_shape[1]],booble)
        gauss_layer_min[int(coord_min[i][0]+offset):int(coord_min[i][0]+offset)+booble_shape[0],
            int(coord_min[i][1]+offset):int(coord_min[i][1]+offset)+booble_shape[1]]=np.where(gauss_layer_min[int(coord_min[i][0]+offset):int(coord_min[i][0]+offset)+booble_shape[0],
            int(coord_min[i][1]+offset):int(coord_min[i][1]+offset)+booble_shape[1]]>booble,
            gauss_layer_min[int(coord_min[i][0]+offset):int(coord_min[i][0]+offset)+booble_shape[0],
            int(coord_min[i][1]+offset):int(coord_min[i][1]+offset)+booble_shape[1]],booble)
        gauss_layer_ave[int(coord_ave[i][0]+offset):int(coord_ave[i][0]+offset)+booble_shape[0],
            int(coord_ave[i][1]+offset):int(coord_ave[i][1]+offset)+booble_shape[1]]=np.where(gauss_layer_ave[int(coord_ave[i][0]+offset):int(coord_ave[i][0]+offset)+booble_shape[0],
            int(coord_ave[i][1]+offset):int(coord_ave[i][1]+offset)+booble_shape[1]]>booble,
            gauss_layer_ave[int(coord_ave[i][0]+offset):int(coord_ave[i][0]+offset)+booble_shape[0],
            int(coord_ave[i][1]+offset):int(coord_ave[i][1]+offset)+booble_shape[1]],booble)
    gauss_layer_max=gauss_layer_max[booble_offset:booble_offset+img.shape[0],booble_offset:booble_offset+img.shape[1]]
    gauss_layer_min=gauss_layer_min[booble_offset:booble_offset+img.shape[0],booble_offset:booble_offset+img.shape[1]]
    gauss_layer_ave=gauss_layer_ave[booble_offset:booble_offset+img.shape[0],booble_offset:booble_offset+img.shape[1]]
    #print 'gauss_layer_max.shape',gauss_layer_max.shape
    #plt.imshow(gauss_layer_max*image_layer, cmap='gray')
#    plt.imshow(gauss_layer_ave, cmap='gray')
    #plt.imshow(img, cmap='gray')
    #print 'len(coord_max)',len(coord_max)

#    for i in reversed(range(len(coord_max))):
#        #print i
#        plt.plot(coord_max[i][1]+offset, coord_max[i][0]+offset, 'r.')
#        plt.plot(coord_ave[i][1]+offset, coord_ave[i][0]+offset, 'g.')
#        plt.plot(coord_min[i][1]+offset, coord_min[i][0]+offset, 'b.')
#        plt.plot([coord_max[i][1]+offset,coord_min[i][1]+offset],
#            [coord_max[i][0]+offset,coord_min[i][0]+offset],'g-')
#        #gauss_layer_max[int(coord_max[i][1]+offset-booble_offset):int(coord_max[i][1]+offset-booble_offset)+booble_shape[1],
#        #    int(coord_max[i][0]+offset-booble_offset):int(coord_max[i][0]+offset-booble_offset)+booble_shape[0]]=booble
#    plt.show()

    layer_num=6-cntr
    max_images[layer_num]=gauss_layer_max*image_layer
    min_images[layer_num]=gauss_layer_min*image_layer
    ave_images[layer_num]=gauss_layer_ave*image_layer
    #plt.imshow(gauss_layer_max)
    #plt.show()
#    #Круги
#    max_images[layer_num]=np.where((gauss_layer_max>0.02)&(gauss_layer_max<0.05),127.,max_images[layer_num])
#    min_images[layer_num]=np.where((gauss_layer_min>0.02)&(gauss_layer_min<0.05),127.,min_images[layer_num])
#    ave_images[layer_num]=np.where((gauss_layer_ave>0.02)&(gauss_layer_ave<0.05),127.,ave_images[layer_num])
for i in list(range(1,6)):
    s=''
    summed_image_max=np.zeros_like(image_layer)
    summed_image_min=np.zeros_like(image_layer)
    summed_image_ave=np.zeros_like(image_layer)
    for j in list(range(i,6)):
        s+=str(j)
        summed_image_max+=max_images[j]
        #summed_image_max=np.where(abs(summed_image_max)>abs(max_images[j]),summed_image_max,max_images[j])
        summed_image_min+=min_images[j]
        #summed_image_min=np.where(abs(summed_image_min)>abs(min_images[j]),summed_image_min,min_images[j])
        summed_image_ave+=ave_images[j]
        #summed_image_ave=np.where(abs(summed_image_ave)>abs(ave_images[j]),summed_image_ave,ave_images[j])
        #print s,len(s)
        #print np.max(summed_image_max)
        cv2.imwrite('./max/'+img_fn+s+'.png',
            mymisc.GammaEncode(np.clip(summed_image_max+ave_brightness, a_min=0., a_max=255.)))
#            mymisc.GammaEncode(cv2.resize(np.clip(summed_image_max+ave_brightness, a_min=0., a_max=255.),
#            (640,640),cv2.INTER_LANCZOS4)))
        cv2.imwrite('./min/'+img_fn+s+'.png',
            mymisc.GammaEncode(np.clip(summed_image_min+ave_brightness, a_min=0., a_max=255.)))
#            mymisc.GammaEncode(cv2.resize(np.clip(summed_image_min+ave_brightness, a_min=0., a_max=255.),
#            (640,640),cv2.INTER_LANCZOS4)))
        cv2.imwrite('./ave/'+img_fn+s+'.png',
            mymisc.GammaEncode(np.clip(summed_image_ave+ave_brightness, a_min=0., a_max=255.)))
#        mymisc.GammaEncode(cv2.resize(np.clip(summed_image_ave+ave_brightness, a_min=0., a_max=255.),
#            (640,640),cv2.INTER_LANCZOS4)))

