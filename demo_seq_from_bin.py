#!/usr/bin/env python
#===============================================================================
# 
# Point cloud sequence viewer with pyqt
#
# Written by Soonmin Hwang (jjang9hsm@gmail.com)
#
#===============================================================================

import sys
from PyQt4 import QtGui, QtCore
from widget.PyGLWidget import PyGLWidget
from OpenGL.GL import *

from plyfile import PlyData
import numpy as np
from OpenGL import GL

import OpenGL.arrays.vbo as glvbo

import numpy as np

#===============================================================================

class SeqGLWidget(PyGLWidget):

    def __init__(self, disp_list, velo_list, isArray=False):

        assert( len(disp_list) == len(velo_list) )

        self.nFrames = len(disp_list)       
        self.fromArray = isArray
        
        self.disp_list = disp_list
        self.velo_list = velo_list

        PyGLWidget.__init__(self)

        self.bShowDisp = True
        self.bShowVelo = True
        self._cur = 0

        self.delta_t = 0.1
        # self.delta_angle = 0.45
        self.delta_angle = 0.3

        # self.delay = 0.01         # 100fps
        self.delay = 0.04           # 25 fps


    # def mouseReleaseEvent(self, _event):
    #     PyGLWidget.mouseReleaseEvent(self, _event)

    #     if _event.button() == QtCore.Qt.RightButton:
    #         self.bShowDisp = not self.bShowDisp

    def keyPressEvent(self, _event):
        import time

        if type(_event) == QtGui.QKeyEvent:
            if _event.key() == QtCore.Qt.Key_P:
                self._cur = max(0, self._cur - 1)
            elif _event.key() == QtCore.Qt.Key_N:
                self._cur = min(self.nFrames-1, self._cur + 1)
            elif _event.key() == QtCore.Qt.Key_R:
                self._cur = 0
                self.setDefaultModelViewMatrix()
            elif _event.key() == QtCore.Qt.Key_S:
                self.bShowDisp = not self.bShowDisp
            elif _event.key() == QtCore.Qt.Key_C:
                import pdb
                pdb.set_trace()                
            elif _event.key() == QtCore.Qt.Key_M:                
                for ii in xrange(self.nFrames):                                        
                    # if ii != 0 and ii % 10 == 0: 
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
        # glRotatef(-10.0, 0.0, 0.0, 1.0)        
        # glTranslated(0.0, 0.0, -10.0)                        
            
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
        # glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearColor(0.0, 0.0, 0.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        self.reset_view()
        
        self.vbo_disp, self.vbo_disp_clr, self.disp_count = self.load_vbo_list(self.disp_list)
        self.vbo_velo, self.vbo_velo_clr, self.velo_count = self.load_vbo_list(self.velo_list)
        
    def load_vbo_list(self, ply_list):        
        vtx_list = [ [] for _ in xrange(self.nFrames) ]
        clr_list = [ [] for _ in xrange(self.nFrames) ]        
        vtx_count = np.zeros( self.nFrames, dtype=np.int32 )     

        for ii, ply in enumerate(ply_list):
            if self.fromArray:                
                vtx_list[ii] = glvbo.VBO( ply['pts'].copy().astype(np.float32) )
                clr_list[ii] = glvbo.VBO( ply['clrs'].copy().astype(np.float32) / 255.0 )
                vtx_count[ii] = len(ply['pts'])
            else:
                pts = np.vstack( [ np.array(list(p)) for p in ply['vertex'] ] ).copy()            
                vtx_list[ii] = glvbo.VBO( pts[:,:3].copy().astype(np.float32) )
                clr_list[ii] = glvbo.VBO( pts[:,3:].copy().astype(np.float32) / 255.0 )
                vtx_count[ii] = len(pts)

        return vtx_list, clr_list, vtx_count
    
#===============================================================================
# Main
#===============================================================================
def readPly(file):
    return PlyData.read(file)


def readBin(file):
    data = np.fromfile(file, dtype=np.float32)        
    N = int(data[0])

    pts = data[1:3*N+1].reshape( (-1,N) ).T
    clrs = data[3*N+1:].reshape( (-1,N) ).T
    return {'pts': pts, 'clrs': clrs}

if __name__ == '__main__':

    from multiprocessing import Pool
    import time

    nFrames = 16

    # tStart = time.time()
    # print 'Load ply files with multiprocessor...',    
    # p = Pool(8)
    # # disp_list = p.map( readPly, [ 'data/0018/%06d_depth.ply' % n for n in xrange(nFrames) ] )
    # # velo_list = p.map( readPly, [ 'data/0018/%06d_velo.ply' % n for n in xrange(nFrames) ] )
    # disp_list = p.map( readBin, [ 'data/0018_bin/%06d_depth.bin' % n for n in xrange(nFrames) ] )
    # velo_list = p.map( readBin, [ 'data/0018_bin/%06d_velo.bin' % n for n in xrange(nFrames) ] )
    # print 'done. (%.2f sec)' % (time.time() - tStart)
    
    disp_list = []
    velo_list = []
    tStart = time.time()
    print 'Load ply files',
    for n in xrange(nFrames):
        if n % 10 == 0:
            print '.',
        # disp_list.append( PlyData.read( '0018/%06d_depth.ply' % n ) )
        # velo_list.append( PlyData.read( '0018/%06d_velo.ply' % n ) )
        disp_list.append( readBin( 'data/0018_bin/%06d_depth.bin' % n ) )
        velo_list.append( readBin( 'data/0018_bin/%06d_velo.bin' % n ) )
    print 'done. (%.2f sec)' % (time.time() - tStart)

    app = QtGui.QApplication(sys.argv)

    tStart = time.time()
    print 'Load points on GPU memory...',
    mainWindow = SeqGLWidget(disp_list, velo_list, True)    
    print 'done. (%.2f sec)' % (time.time() - tStart)

    
    mainWindow.resize(1280, 960)


    tStart = time.time()
    print 'Launch window...',
    mainWindow.show()
    print 'done. (%.2f sec)' % (time.time() - tStart)

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
