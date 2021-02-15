#!/usr/bin/python3
#import os,sys,numpy,hashlib,time
import subprocess,shlex,multiprocessing,glob,os
#from PIL import Image
from multiprocessing.pool import ThreadPool

#Tnx 2 dano https://stackoverflow.com/users/2073595/dano
#https://stackoverflow.com/questions/25120363/multiprocessing-execute-external-command-and-wait-before-proceeding
#cmd -- строка с командой оболочки
def call_proc(cmd):
    """ This runs in a separate thread. """
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    #print "Done", p.pid
    return (out, err)

inpdir='./исходные1280x1280/'
#print(inpdir.encode())
#print(inpdir.encode().decode())
#exit(0)
files=[os.path.basename(fn) for fn in glob.glob(inpdir+'*.png')]
pool = ThreadPool(multiprocessing.cpu_count())
results = []
for fname in files:
    cmd='./model_alpha_cpu.py '+fname
    #print cmd
    results.append(pool.apply_async(call_proc, (cmd,)))
pool.close()
pool.join()
for result in results:
    out, err = result.get()
    print("out: {} err: {}".format(out, err))
