#!/usr/bin/python3
# -*- coding: utf-8 -*-
#^    For Python 2    ^
import cv2,os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import distance_matrix
import mymisc
from skimage import exposure
#Каталог с исходными изображениями / Folder with source images
srcpath='./исходные/'
#Каталог для результатов преобразований / Destination folder
dstpath='./исходные1280x1280/'
#В гамма-декодированных величинах / Luminance in gamma decoded values
backcolor=mymisc.GammaDecode(127) #55. #127.5
avecolor=mymisc.GammaDecode(127) #55. #127.5
#Среднеквадратический контраст / RMS-contrast
rms=0.45
os.system('mkdir '+dstpath)
output_size=1280
files=os.listdir(srcpath)
for fname in files:
    print(fname)
    im = cv2.imread(srcpath+fname, cv2.IMREAD_UNCHANGED)
    #Сохраняем маску прозрачности / Storing the transparency mask
    mask=im[:,:,3]
    #В оттенки серого с 4-хбайтной реальной точностью / Converting to float32 coded grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    im_gray = mymisc.GammaDecode(im_gray)
    #print(type(im_gray[0,0]))
    #exit(0)
    #Далее определяем размеры объекта в изображении
    im_x,im_y=mask.shape
    print('im_x,im_y',im_x,im_y)
    tmpmask=np.zeros_like(mask)
    #Уменьшаем маску, чтобы по разности исходной и зуммированной получить контур
    #Если считать матрицу расстояний между всеми точками, будет либо очень медленно,
    #либо потребуется оочень много памяти -- для distance_matrix 64 Гб не хватает
    tmp_zoomed=zoom(mask,(np.min([im_x,im_y])-2)/float(np.min([im_x,im_y])))
    #print(tmp_zoomed.shape)
    #print ((im_x-2)/float(im_x),(im_y-2)/float(im_y))
    #Зуммированная маска помещается в массив, равный по размерам mask
    tmpmask[1:1+tmp_zoomed.shape[0],1:1+tmp_zoomed.shape[1]]=tmp_zoomed
    #Определяем разницу масок, получаем пиксели по контуру
    contour_mask=np.logical_xor(mask,tmpmask)
    #plt.imshow(contour_mask,cmap='gray')
    #plt.show()
    #Координаты замаскированных точек
    coords=np.argwhere(contour_mask).astype(np.float32)#.tolist()
    #Выбираем максимальное расстояние из расстояний между всеми точками в окрестностях контура
    obj_size=np.max(distance_matrix(coords,coords))
    print(obj_size)
    #Теперь мы знаем максимальный размер объекта и можем определить масштабирующий коэффициент
    zoom_coef=880./obj_size

    bool_mask=mask.astype('bool')
    #По примеру из PsychoPy
    # convert to mean zero and specified RMS contrast
    im_gray = np.where(bool_mask,im_gray - np.mean(im_gray[bool_mask]),im_gray)
    im_gray = im_gray / np.std(im_gray[bool_mask])
    im_gray = im_gray * rms

    #Подтянем яркость, перейдём к градациям серого и закрасим фон
    im_gray=np.where(bool_mask,im_gray*avecolor+avecolor,backcolor)
    #print('np.min(im_gray[bool_mask])',np.min(im_gray[bool_mask]))
    #print('np.max(im_gray[bool_mask])',np.max(im_gray[bool_mask]))
    im_gray = np.clip(im_gray, a_min=0., a_max=255.)
    im_gray=mymisc.GammaEncode(im_gray)
    #if np.isnan(np.mean(im_gray[bool_mask])):
    # print('КУКУКУ')
    # os.system('echo '+fname+' >> resize_errors.csv')
    print(np.mean(im_gray[bool_mask]),np.min(im_gray[bool_mask]),np.max(im_gray[bool_mask]))
    tmp=mymisc.GammaDecode(im_gray[bool_mask])
    print('rms',np.std(tmp)/np.mean(tmp))

    #Теперь можно масштабировать. Будем делать это с использованием cv2, так там есть кошерная интерполяция
    im_zoomed=cv2.resize(im_gray,(np.round(im_y*zoom_coef).astype('int'),np.round(im_x*zoom_coef).astype('int')),
     interpolation=cv2.INTER_LANCZOS4)
    x_zoomed,y_zoomed=im_zoomed.shape
    print('im_zoomed.shape',(x_zoomed,y_zoomed))
    mask_zoomed=cv2.resize(mask,(np.round(im_y*zoom_coef).astype('int'),np.round(im_x*zoom_coef).astype('int')),
     interpolation=cv2.INTER_NEAREST)

    #Если картинка получилась очень большой (обычно для картинок из старого набора)
    if x_zoomed>output_size:
     zoom_diff=x_zoomed-output_size
     im_zoomed=im_zoomed[zoom_diff//2:x_zoomed-zoom_diff//2-1,:]
     mask_zoomed=mask_zoomed[zoom_diff//2:x_zoomed-zoom_diff//2-1,:]
    if y_zoomed>output_size:
     zoom_diff=x_zoomed-output_size
     im_zoomed=im_zoomed[:,zoom_diff//2:x_zoomed-zoom_diff//2-1]
     mask_zoomed=mask_zoomed[:,zoom_diff//2:x_zoomed-zoom_diff//2-1]
    x_zoomed,y_zoomed=im_zoomed.shape
    print('im_zoomed.shape',(x_zoomed,y_zoomed))

    upper_x_coord=int(output_size-x_zoomed)//2
    upper_y_coord=int(output_size-y_zoomed)//2
    print('upper_x_coord,upper_y_coord',upper_x_coord,upper_y_coord)
    #Вставляем результат в большой массив (1280x1280)
    output_image=np.full((output_size,output_size),mymisc.GammaEncode(backcolor))
    output_image[upper_x_coord:upper_x_coord+x_zoomed,upper_y_coord:upper_y_coord+y_zoomed]=im_zoomed
    output_mask=np.full((output_size,output_size),0)
    output_mask[upper_x_coord:upper_x_coord+x_zoomed,upper_y_coord:upper_y_coord+y_zoomed]=mask_zoomed
    #Пишем на диск
    cv2.imwrite(dstpath+fname,output_image)
    cv2.imwrite('./masks/'+fname,output_mask)
    #plt.imshow(output_image,cmap='gray')
    #plt.show()
