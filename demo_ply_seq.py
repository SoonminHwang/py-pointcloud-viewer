#!/usr/bin/env python
#===============================================================================
# 
# Point cloud sequence viewer with pyqt
#
# Written by Soonmin Hwang (jjang9hsm@gmail.com)
#
#===============================================================================
from __future__ import print_function

import sys, os
curpath = os.path.dirname(__file__)
plypath = os.path.join(curpath, 'python-plyfile')
if plypath not in sys.path:
    sys.path.insert(0, plypath)
from plyfile import PlyData

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage
from widget.PyGLWidget import PyGLWidget
from OpenGL.GL import *
# from OpenGL.GL import shaders
# from OpenGL import GL
import OpenGL.arrays.vbo as glvbo

import numpy as np
from multiprocessing import Pool
import time
import cv2
import datetime

#===============================================================================
BACKGROUND_BLACK = True
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960

DELAY_FOR_25FPS = 0.04
DELAY_FOR_100FPS = 0.01

def qimage2numpy(qimage, dtype = 'array'):
    """Convert QImage to numpy.ndarray.  The dtype defaults to uint8
    for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
    for 32bit color images.  You can pass a different dtype to use, or
    'array' to get a 3D uint8 array for color images."""
    result_shape = (qimage.height(), qimage.width())
    temp_shape = (qimage.height(),
                  int(qimage.bytesPerLine() * 8 / qimage.depth()) )
    if qimage.format() in (QImage.Format_ARGB32_Premultiplied,
                           QImage.Format_ARGB32,
                           QImage.Format_RGB32):
        if dtype == 'rec':
            dtype = bgra_dtype
        elif dtype == 'array':
            dtype = np.uint8
            result_shape += (4, )
            temp_shape += (4, )
    elif qimage.format() == QImage.Format_Indexed8:
        dtype = np.uint8
    else:
        raise ValueError("qimage2numpy only supports 32bit and 8bit images")
    # FIXME: raise error if alignment does not match
    buf = qimage.bits().asstring(qimage.bytesPerLine()*qimage.height())
    result = np.frombuffer(buf, dtype).reshape(temp_shape)
    if result_shape != temp_shape:
        result = result[:,:result_shape[1]]
    if qimage.format() == QImage.Format_RGB32 and dtype == np.uint8:
        result = result[...,:3]
    return result


class SeqGLWidget(PyGLWidget):

    def __init__(self, disp_list, velo_list):

        PyGLWidget.__init__(self)

        assert( len(disp_list) == len(velo_list) )

        self.nFrames = len(disp_list)       
        
        self.disp_list = disp_list
        self.velo_list = velo_list

        
        self.bShowDisp = True
        self.bShowVelo = True
        self._cur = 0

        self.delta_t = 0.1
        # self.delta_angle = 0.45
        self.delta_angle = 0.3

        # self.delay = 0.01         # 100fps
        self.delay = DELAY_FOR_25FPS           # 25 fps


    def keyPressEvent(self, _event):

        if type(_event) == QtGui.QKeyEvent:
            if _event.key() == QtCore.Qt.Key_P:
                self._cur = max(0, self._cur - 1)

            elif _event.key() == QtCore.Qt.Key_N:
                self._cur = min(self.nFrames-1, self._cur + 1)

            elif _event.key() == QtCore.Qt.Key_R:
                self._cur = 0
                self.setDefaultModelViewMatrix()

            elif _event.key() == QtCore.Qt.Key_T:
                self.bShowDisp = not self.bShowDisp

            elif _event.key() == QtCore.Qt.Key_D:
                img = self.grabFrameBuffer()
                print( type( qimage2numpy(img) ) )

            elif _event.key() == QtCore.Qt.Key_S:
                
                cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')                
                vid_file = cur_time + '.avi'
                # capture_dir = os.path.join('screenshot', cur_time)                
                # if os.path.exists(capture_dir):
                #     os.makedirs(capture_dir)


                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(vid_file,fourcc, 20.0, (WINDOW_WIDTH, WINDOW_HEIGHT))

                for ii in range(self.nFrames):
                    # if ii != 0 and ii % 10 == 0: 

                    ### Backward direction
                    # if self.nFrames-1-ii < 0:
                    #     break
                    # self._cur = max(0, self.nFrames-1-ii)
                    
                    self._cur = ii

                    self.translate([self.delta_t, 0.0, 0.0])                        
                    self.rotate([0.0, 1.0, 1.0], self.delta_angle)                                    
                    self.updateGL()       

                    img = self.grabFrameBuffer()
                    out.write(qimage2numpy(img))

                    # img.save( os.path.join(capture_dir, 'screenshot_%09d.jpg' % ii ), "JPG")

                    time.sleep(self.delay)

                out.release()

                print('Save to {:s}'.format(vid_file))

            elif _event.key() == QtCore.Qt.Key_M:                
                for ii in range(self.nFrames):
                    self._cur = min(self.nFrames-1, self._cur+1)

                    self.translate([self.delta_t, 0.0, 0.0])                        
                    self.rotate([0.0, 1.0, 1.0], self.delta_angle)                                    
                    self.updateGL()                    
                    time.sleep(self.delay)


    def setDefaultModelViewMatrix(self):
        self.modelview_matrix_ = np.array(  [[ 2.16917381e-01,   3.58195961e-01,  -9.08098578e-01,   0.00000000e+00],
                                             [-9.74002719e-01,   1.71728823e-02,  -2.25885659e-01,   0.00000000e+00],
                                             [-6.53166994e-02,   9.33488667e-01,   3.52608502e-01,   0.00000000e+00],
                                             [ 2.95688105e+00,   6.45137596e+00,  -4.45000000e+01,   1.00000000e+00]] )

    def paintGL(self):        

        PyGLWidget.paintGL(self)
        
        glPushMatrix()        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        vtx_velo = self.vbo_velo[self._cur]
        clr_velo = self.vbo_velo_clr[self._cur]
        cnt_velo = self.velo_count[self._cur]

        vtx_disp = self.vbo_disp[self._cur]
        clr_disp = self.vbo_disp_clr[self._cur]
        cnt_disp = self.disp_count[self._cur]
        
        vtx_velo.bind()
        glVertexPointer(3, GL_FLOAT, 0, vtx_velo)
        vtx_velo.unbind()
        
        clr_velo.bind()
        glColorPointer(3, GL_FLOAT, 0, clr_velo)
        clr_velo.unbind()

        glDrawArrays(GL_POINTS, 0, cnt_velo)
       
        vtx_disp.bind()
        glVertexPointer(3, GL_FLOAT, 0, vtx_disp)
        vtx_disp.unbind()
                
        clr_disp.bind()
        glColorPointer(3, GL_FLOAT, 0, clr_disp)
        clr_disp.unbind()
        
        glDrawArrays(GL_POINTS, 0, cnt_disp)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        self.renderText( 0, 0, 0, 'Cur: %d' % self._cur )
        glPopMatrix()

    def initializeGL(self):
        # OpenGL state        
        if BACKGROUND_BLACK:
            glClearColor(0.0, 0.0, 0.0, 0.0)
        else:
            glClearColor(1.0, 1.0, 1.0, 1.0)

        glEnable(GL_DEPTH_TEST)
        self.reset_view()
        
        self.vbo_disp, self.vbo_disp_clr, self.disp_count = self.load_vbo_list(self.disp_list)
        self.vbo_velo, self.vbo_velo_clr, self.velo_count = self.load_vbo_list(self.velo_list)

        
    def load_vbo_list(self, ply_list):        
        vtx_list = [ [] for _ in range(self.nFrames) ]
        clr_list = [ [] for _ in range(self.nFrames) ]        
        vtx_count = np.zeros( self.nFrames, dtype=np.int32 )     

        for ii, ply in enumerate(ply_list):
            pts = np.vstack( [ np.array(list(p)) for p in ply['vertex'] ] ).copy()            
            vtx_list[ii] = glvbo.VBO( pts[:,:3].copy().astype(np.float32) )
            clr_list[ii] = glvbo.VBO( pts[:,3:].copy().astype(np.float32) / 255.0 )
            vtx_count[ii] = len(pts)

        return vtx_list, clr_list, vtx_count
    
#===============================================================================
# Main
#===============================================================================
# def readPly(file):
#     return PlyData.read(file)

def readPly(file):
    '''
        Since the first 10 lines of the ply file (header) is as following:
            ply
            format ascii 1.0
            element vertex 126428
            property double x
            property double y
            property double z
            property uchar red
            property uchar green
            property uchar blue
            end_header

        So, just skip to read these.
    '''
    pts_with_colors = np.loadtxt( file, skiprows=10, ndmin=2 )
    return dict(vertex=pts_with_colors)

def readBin(file):
    data = np.fromfile(file, dtype=np.float32)        
    N = int(data[0])
    pts = data[1:3*N+1].reshape( (-1,N) ).T
    clrs = data[3*N+1:].reshape( (-1,N) ).T    
    return dict(vertex=np.hstack((pts, clrs)))

if __name__ == '__main__':


    app = QtWidgets.QApplication(sys.argv)

    filter = "Ply (*.ply); Binary (*.bin)"
    file_name = QtWidgets.QFileDialog()
    file_name.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
    
    lidar_list = file_name.getOpenFileNames(None, "Select files for LiDAR", ".", filter)[0]
    depth_list = file_name.getOpenFileNames(None, "Select files for Depth", ".", filter)[0]

    assert len(lidar_list) > 0 and len(lidar_list) == len(depth_list)    

    readFunc = readPly if lidar_list[0].endswith('ply') else readBin


    tStart = time.time()
    print('Load ply files with multiprocessor...', end='')    
    p = Pool(4)    
    disp_list = p.map( readFunc, [ str(file) for file in lidar_list ] )
    velo_list = p.map( readFunc, [ str(file) for file in depth_list ] )    
    # disp_list = [ readPly( str(file) ) for file in lidar_list ]
    # velo_list = [ readPly( str(file) ) for file in depth_list ]
    print('done. (%.2f sec)' % (time.time() - tStart))
    


    tStart = time.time()
    print('Load points on GPU memory...', end='')
    mainWindow = SeqGLWidget(disp_list, velo_list)    
    print('done. (%.2f sec)' % (time.time() - tStart))
    
    # mainWindow.resize(1280, 960)
    mainWindow.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    tStart = time.time()
    print('Launch window...', end='')
    mainWindow.show()
    print('done. (%.2f sec)' % (time.time() - tStart))

    mainWindow.setDefaultModelViewMatrix()

    mainWindow.raise_() # Need this at least on OS X, otherwise the window ends up in background
    sys.exit(app.exec_())

#===============================================================================
#
# Local Variables:
# mode: Python
# indent-tabs-mode: nil
# End:
#
#===============================================================================
