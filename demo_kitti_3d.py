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
from OpenGL import GL

import OpenGL.arrays.vbo as glvbo

import numpy as np
import pykitti.raw as raw

#===============================================================================
# from xml.etree.ElementTree import parse
import parseTrackletXML as xmlParser

import os

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}


edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

class kitti_object(raw):
    def __init__(self, base_path, data, drive, **kwargs):
        super(raw, self).__init__(base_path, data, drive, kwargs)
        """Set the path and pre-load calibration data and timestamps."""
        
        # base_path should include 'training' or 'testing'        
        assert 'training' in base_path or 'testing' in base_path

        self.base_path = base_path

        self.calib_path = os.path.join(base_path, 'calib')
                
        self.cam2_path = os.path.join(base_path, 'calib')
        self.cam3_path = os.path.join(base_path, 'calib')
        

        self.subset = kwargs.get('subset', 'train')




class kitti_raw(raw):
    # def __init__(self, base_path, data, drive, **kwargs):
    #     super(raw, self).__init__(base_path, data, drive, kwargs)

    def load_tracklets_for_frames(self):
        """
        Loads dataset labels also referred to as tracklets, saving them individually for each frame.
        (https://github.com/navoshta/KITTI-Dataset/blob/master/kitti-dataset.ipynb)

        Parameters
        ----------
        n_frames    : Number of frames in the dataset.
        xml_path    : Path to the tracklets XML.

        Returns
        -------
        Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
        contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
        types as strings.
        """
        xml_path = os.path.join(self.data_path, 'tracklet_labels.xml')
        tracklets = xmlParser.parseXML(xml_path)

        frame_tracklets = {}
        frame_tracklets_types = {}
        for i in range(len(self)):
            frame_tracklets[i] = []
            frame_tracklets_types[i] = []

        # loop over tracklets
        for i, tracklet in enumerate(tracklets):
            # this part is inspired by kitti object development kit matlab code: computeBox3D
            h, w, l = tracklet.size
            # in velodyne coordinates around zero point and without orientation yet
            trackletBox = np.array([
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]
            ])

            # import pdb
            # pdb.set_trace()

            # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
                # determine if object is in the image; otherwise continue
                if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue
                # re-create 3D bounding box in velodyne coordinate system
                yaw = rotation[2]  # other rotations are supposedly 0
                assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                rotMat = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]
                ])
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
                frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                    tracklet.objectType]

        self.frame_tracklets = frame_tracklets
        self.frame_tracklets_types = frame_tracklets_types

        # return (frame_tracklets, frame_tracklets_types)





class SeqGLWidget(PyGLWidget):

    def __init__(self, dataset):

        # assert( len(disp_list) == len(velo_list) )

        # self.nFrames = len(disp_list)       
        self.nFrames = len(dataset)
        self.dataset = dataset
        
        self.velo = self.dataset.get_velo(0)
        # self.disp_list = disp_list
        # self.velo_list = velo_list
        # self.velo = dataset.get_velo(2)
        # self.iter = iter(dataset)

        PyGLWidget.__init__(self)

        self.bShowDisp = True
        self.bShowVelo = True
        self._cur = 0

        self.delta_t = 0.1
        # self.delta_angle = 0.45
        self.delta_angle = 0.3

        # self.delay = 0.01         # 100fps
        self.delay = 0.04           # 25 fps
        # self.delay = 1         # 10fps


    # def mouseReleaseEvent(self, _event):
    #     PyGLWidget.mouseReleaseEvent(self, _event)

    #     if _event.button() == QtCore.Qt.RightButton:
    #         self.bShowDisp = not self.bShowDisp

    def load_data(self, ii):

        velo = self.dataset.get_velo(ii)                    
        self.velo_vbo = glvbo.VBO( velo[:,:3].copy().astype(np.float32) )
        self.velo_count = len(velo)

        
        velo = velo.T
        velo[-1,:] = 1
    
        point_cam2 = self.dataset.calib.T_cam2_velo.dot(velo)
        point_cam2 = point_cam2 / point_cam2[-1,:]
        point_cam2_proj = self.dataset.calib.P_rect_20.dot( point_cam2 )
        point_cam2_proj = point_cam2_proj / point_cam2_proj[-1,:]

        image = np.array( self.dataset.get_cam2(ii) )
        height, width = image.shape[:2]

        valid = (point_cam2_proj[0,:] > 0) * (point_cam2_proj[0,:] < width) * \
                (point_cam2_proj[1,:] > 0) * (point_cam2_proj[1,:] < height) * \
                (velo[0,:] > 0)


        colors = 255*np.ones( (velo.shape[1], 3), dtype=np.float32 )

        x = point_cam2_proj[0:1, valid].astype(np.uint16)
        y = point_cam2_proj[1:2, valid].astype(np.uint16)
        
        b = image[y, x, 0]
        g = image[y, x, 1]
        r = image[y, x, 2]

        colors[valid,0] = b
        colors[valid,1] = g
        colors[valid,2] = r

        # self.clr_count = len(b)
        self.clr_vbo = glvbo.VBO( colors.copy() / 255.0 )


        self.boxes = [ box.astype(np.float32) for box in self.dataset.frame_tracklets[ii] ]
        self.box_count = len(self.boxes)
        # objs = np.concatenate(self.dataset.frame_tracklets[ii], axis=1).T
        # self.obj_vbo = glvbo.VBO( objs.copy() )
        # self.obj_count = len(objs)
        # objclr_vbo = 255*np.ones( (len(objs), 3), dtype=np.float32 )
        # objclr_vbo[:,2] = 255.0
        # self.objclr_vbo = glvbo.VBO( objclr_vbo.copy() / 255.0 )


        
    def draw_cube(self, ii):
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3f( self.boxes[ii][0,vertex], self.boxes[ii][1,vertex], self.boxes[ii][2,vertex] )
        glEnd()

    def keyPressEvent(self, _event):
        import time

        if type(_event) == QtGui.QKeyEvent:
            if _event.key() == QtCore.Qt.Key_P:
                self._cur = max(0, self._cur - 1)
                self.load_data(self._cur)

            elif _event.key() == QtCore.Qt.Key_N:
                self._cur = min(self.nFrames-1, self._cur + 1)
                self.load_data(self._cur)

            elif _event.key() == QtCore.Qt.Key_R:
                self._cur = 0
                self.load_data(self._cur)
                self.setDefaultModelViewMatrix()

            elif _event.key() == QtCore.Qt.Key_M:                
                
                for ii in range( self._cur, len(self.dataset) ):                
                    self.load_data(ii)
                    # self.translate([self.delta_t, 0.0, 0.0])                        
                    # self.rotate([0.0, 1.0, 1.0], self.delta_angle)                                    
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

        vtx_velo = self.velo_vbo
        clr_velo = self.clr_vbo
        cnt_velo = self.velo_count
            
        vtx_velo.bind()
        glVertexPointer(3, GL_FLOAT, 0, vtx_velo)
        vtx_velo.unbind()
        
        clr_velo.bind()
        glColorPointer(3, GL_FLOAT, 0, clr_velo)
        clr_velo.unbind()

        glDrawArrays(GL_POINTS, 0, cnt_velo)
       
        # vtx_disp.bind()
        # glVertexPointer(3, GL_FLOAT, 0, vtx_disp)
        # vtx_disp.unbind()
                
        # clr_disp.bind()
        # glColorPointer(3, GL_FLOAT, 0, clr_disp)
        # clr_disp.unbind()
        
        # glDrawArrays(GL_POINTS, 0, cnt_disp)

        # glDisableClientState(GL_VERTEX_ARRAY)
        # glDisableClientState(GL_COLOR_ARRAY)

        glColor3f(0.0,1.0,0.0)
        for ii in range(self.box_count):
            self.draw_cube(ii)


        # glEnableClientState(GL_VERTEX_ARRAY)
        # glEnableClientState(GL_COLOR_ARRAY)

        # obj = self.obj_vbo
        # clr = self.objclr_vbo
        # cnt = self.obj_count
            
        # obj.bind()
        # glVertexPointer(3, GL_FLOAT, 0, obj)
        # obj.unbind()
        
        # clr.bind()
        # glColorPointer(3, GL_FLOAT, 0, clr)
        # clr.unbind()

        # glDrawArrays(GL_POINTS, 0, cnt)
       
        # vtx_disp.bind()
        # glVertexPointer(3, GL_FLOAT, 0, vtx_disp)
        # vtx_disp.unbind()
                
        # clr_disp.bind()
        # glColorPointer(3, GL_FLOAT, 0, clr_disp)
        # clr_disp.unbind()
        
        # glDrawArrays(GL_POINTS, 0, cnt_disp)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # self.renderText( 0, 0, 0, 'Cur: %d' % self._cur )
        glPopMatrix()

        

    def initializeGL(self):
        # OpenGL state
        # glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearColor(0.0, 0.0, 0.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        self.reset_view()
                    

        # self.velo_count = len(self.velo)
        # self.velo_vbo = glvbo.VBO( self.velo[:,:3].astype(np.float32) )

        self.load_data(self._cur)
        

    
#===============================================================================
# Main
#===============================================================================

if __name__ == '__main__':

    import time

    basedir = 'E:/Dataset/KITTI/Raw/data'

    date = '2011_09_26'
    drive = '0005'

    # dataset = pykitti.raw(basedir, date, drive, frames=range(0, 100))
    # dataset = kitti_raw(basedir, date, drive, frames=range(0, 100))
    dataset = kitti_raw(basedir, date, drive)

    dataset.load_tracklets_for_frames()
    

    
    for ii in range(len(dataset)):
        image = dataset.get_cam2(ii)
        width, height = image.size

        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)

        boxes = dataset.frame_tracklets[ii]

        for box in boxes:
            box = np.concatenate( (box, np.ones((1,8)) ), axis=0)
            
            # point_cam2 = dataset.calib.T_cam2_velo.dot(box)
            # point_cam2 = point_cam2 / point_cam2[-1,:]
            # point_cam2_proj = dataset.calib.P_rect_20.dot( point_cam2 )

            mat = dataset.calib.P_rect_20.dot(dataset.calib.T_cam2_velo)
            point_cam2_proj = mat.dot( box )            
            point_cam2_proj = point_cam2_proj / point_cam2_proj[-1,:]
        
            for edge in edges:
                draw.line( tuple(point_cam2_proj[:2, edge[0]]) + tuple(point_cam2_proj[:2, edge[1]]) )

        image.save( 'test/frame_{:03d}.png'.format(ii) )


    import pdb
    pdb.set_trace()

    # velo = dataset.get_velo(0).T
    # velo[-1,:] = 1
    # point_cam2 = dataset.calib.T_cam2_velo.dot(velo)
    # point_cam2 = point_cam2 / point_cam2[-1,:]
    # point_cam2_proj = dataset.calib.P_rect_20.dot( point_cam2 )
    # point_cam2_proj = point_cam2_proj / point_cam2_proj[-1,:]

    # image = dataset.get_cam2(0)
    # width, height = image.size

    # valid = (point_cam2_proj[0,:] > 0) * (point_cam2_proj[0,:] < width) * \
    #         (point_cam2_proj[1,:] > 0) * (point_cam2_proj[1,:] < height) * \
    #         (velo[0,:] > 0)

    # import numpy as np
    # depth = np.zeros( (height, width) )

    # from scipy.sparse import csr_matrix

    # x = point_cam2_proj[0:1, valid].astype(np.uint16)
    # y = point_cam2_proj[1:2, valid].astype(np.uint16)
    # d = velo[0:1, valid].astype(np.uint8)
    # depth[y, x] = 2*d

    # import cv2
    # cv2.imwrite( 'depth.png', depth )
    # image.save( 'image.png' )

    # image = np.array(image)
    # # image[y, x, np.ones_like(x)] = d
    # # image[y, x, 2*np.ones_like(x)] = d
    # # image[y, x, 0*np.ones_like(x)] = d
    # # cv2.imwrite( 'd+image.png', image[:,:,(2,1,0)])
    



    # dcolor = np.zeros_like(image)
    # dcolor[:,:,1] = 32
    # dcolor[y, x, 0] = image[y, x, 0]
    # dcolor[y, x, 1] = image[y, x, 1]
    # dcolor[y, x, 2] = image[y, x, 2]

    # cv2.imwrite( 'd+image.png', dcolor[:,:,(2,1,0)])
 
    app = QtGui.QApplication(sys.argv)

    tStart = time.time()
    print 'Load points on GPU memory...',
    mainWindow = SeqGLWidget(dataset)
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
# 