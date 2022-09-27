# -*- coding: utf-8 -*-
'''
@author: alienor,david
'''

from PIL import Image
import requests
from io import BytesIO
import os
import urllib.request

''' Before running this code, the virtual scanner should be initiated following 
these instructions: https://github.com/romi/blender_virtual_scanner. The scanner is hosted on localhost:5000'''



class virtual_scan():
    
    def __init__(self, w = None, h = None, f = None, path=None):
        self.R = 55 #radial distance from [x, y] center
        self.N = 72 #number of positions on the circle
        self.z = 50 #camera elevation
        self.rx = 60# camera tilt
        self.ry = 0 #camera twist
        self.w = 1920 #horizontal resolution
        self.h = 1080 #vertical resolution
        self.f = 24 #focal length in mm
        
        self.localhost = "http://localhost:5000/" 
        if (path==None): self.path = 'data/scans'         
        else: self.path=path
            
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        if f is None:
            f = self.f
        
        #CAMERA API blender
        url_part = 'camera?w=%d&h=%d&f=%d'%(w, h, f)
        contents = urllib.request.urlopen(self.localhost + url_part).read()
        
    def create(self,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
    
    #def load_im(self,num):
        '''This function loads the arabidopsis mesh. It should be included in the data folder associated to the virtual scanner as a .obj '''
        '''url_part = "load/arabidopsis_%s"%num
        contents = urllib.request.urlopen(self.localhost + url_part).read()
        return contents'''

    def load_im(self, path):
        url_part = "load/" + path 
        contents = urllib.request.urlopen(self.localhost + url_part).read()
        return contents
    
    def render(self, x, y, z, rx, ry, rz):
        '''This functions calls the virtual scanner and loads an image of the 3D mesh taken from 
        a virtual camera as position x, y, z and orientations rx, ry, rz'''
        url_part = "move?x=%s&y=%s&z=%s&rx=%s&ry=%s&rz=%s"%(x, y, z, rx, ry, rz)
        contents = urllib.request.urlopen(self.localhost + url_part).read()
        response = requests.get(self.localhost + 'render')
        img = Image.open(BytesIO(response.content))
        
        return img   

    def close():
        pass


