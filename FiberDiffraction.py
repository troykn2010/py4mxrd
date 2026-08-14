import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from .detector import detector

class fiber_image():
    def __init__(self,image,mask,centeri=0,centerj=0,align_threshold=15,AutoCentering=False,quiet=True,phi = 0):
        #image is a 2d nd.array       
        self.image = image
        self.mask = mask
        self.align_threshold = align_threshold
        self.phi = phi
        self.quiet = quiet
        self.warning = False
        self.AutoCentering = AutoCentering

        #Initial processing
        imagethres = self.mask*(self.image>self.align_threshold)
        imagethres = imagethres.astype(np.float32)
        self.moments = cv2.moments(imagethres)
        
        if self.AutoCentering:
            self.centerx = self.moments['m10']/self.moments['m00']
            self.centery = self.moments['m01']/self.moments['m00']
            self.centerj = self.centerx
            self.centeri = self.centery    

            if not self.quiet:
                print((self.centerx,self.centery))
        else:
            #keep track of centers on both xy and ij indexing
            self.centeri = centeri
            self.centerj = centerj
            self.centerx = centerj
            self.centery = centeri
    
    @staticmethod
    def compute_moments(image):
        #Image moments. #cv2 doesn't take in int32 so convert to float
        return cv2.moments(image.astype(np.float32))
    
    @staticmethod
    def rotate_image(image,center, angle):
        #cv2 xy indexing
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    @staticmethod
    def CenterByPadding(image,centeri,centerj,subpixel=True,interp='Default'):
        #Enforces NxM where N and M odd. So N//2 and M//2 is always at center of image

        #numpy ij indexing
        l = image.shape[0]
        m = image.shape[1]

        #Move to center by zeropadding image
        lm1 = l-1
        int_centeri = int(centeri)
        int_centerj = int(centerj)
        if int_centeri >= l//2:
            pad0 = (0,2*int_centeri-lm1)
        elif centeri < l//2:
            pad0 = (lm1-2*int_centeri,0)

        mm1 = m-1
        if int_centerj >= m//2:
            pad1 = (0,2*int_centerj-mm1)
        elif int_centerj < m//2:
            pad1 = (mm1-2*int_centerj,0)
        image = np.pad(image, (pad0,pad1)) 


        if subpixel:
            #Subpixel shift to corner of pixel 
            if interp == 'Default':
                interp = cv2.INTER_LINEAR
            # image = np.pad(image, ( (1,1),(1,1))) #pad all sides by 1 pixel to avoid info loss  #Just doesn't matter enough for large images
            l2 = image.shape[0]
            m2 = image.shape[1]

            ishift = centeri % 1
            jshift = centerj % 1
            # cv2 translation matrix
            translation_matrix = np.array([
                [1, 0, -jshift],
                [0, 1, -ishift]
            ], dtype=np.float32)
            image = cv2.warpAffine(src=image,
                    M=translation_matrix,
                    dsize=(m2, l2),
                    flags = interp)

        # return image
        return image    

    def AutoAlign(self):
        """ Orientate images based on image moments. NB: cv2's xy is flipped from numpy's ij
        """
        moments = self.moments
        if moments['mu20'] == moments['mu02']:
            phi = 0
        else:
            phi = np.arctan(2*moments['mu11']/(moments['mu20']-moments['mu02']))/2
        w,v = np.linalg.eig(np.array([ [moments['mu20'],moments['mu11']] , [moments['mu11'],moments['mu02']]  ]))        
       
        i = abs(w).argmax()
        if abs(v[i,0])<abs(v[i,1]):
            phi = np.pi/2 + phi
                
        self.image = self.rotate_image(self.image.astype(np.float32),(self.centerx,self.centery),phi*180/np.pi)
        self.mask = self.rotate_image(self.mask.astype(np.uint8),(self.centerx,self.centery),phi*180/np.pi)
        self.phi = phi #Keep track of image rotation
        
    def RotateAndApplySymmetry(self,align='auto',subpixel=True):
        if subpixel:
            self.image = self.CenterByPadding(self.image.astype(np.float32),self.centeri,self.centerj,subpixel=True)
            self.mask = self.CenterByPadding(self.mask.astype(np.uint8),self.centeri,self.centerj,subpixel=True,interp = cv2.INTER_NEAREST)
            kernel = np.ones((3,3),dtype = np.uint8)
            self.mask = cv2.erode(self.mask,kernel)
            self.image = self.image*self.mask
        else:
            self.image = self.CenterByPadding(self.image*self.mask,self.centeri,self.centerj,subpixel=False)
            self.mask = self.CenterByPadding(self.mask,self.centeri,self.centerj,subpixel=False)
        #center is now in the middle of the image
        self.centeri = self.image.shape[0]//2
        self.centerj = self.image.shape[1]//2
        self.centerx = self.centerj
        self.centery = self.centeri
        
        if align == 'auto':
            self.AutoAlign()
        elif align == 'manual':
            rotationcenter = (self.centerx,self.centery)
            self.image = self.rotate_image(self.image.astype(np.float32),rotationcenter,self.phi*180/np.pi)
            self.mask = self.rotate_image(self.mask.astype(np.uint8),rotationcenter,self.phi*180/np.pi)
            
        mask2 = np.copy(self.mask).astype(np.float32)
        mask2 += np.flipud(self.mask)
        mask2 += np.fliplr(self.mask)
        mask2 += np.flipud(np.fliplr(self.mask))
        mask2[mask2==0] = np.inf

        output = np.copy(self.image).astype(np.float32)
        output += np.flipud(self.image)
        output += np.fliplr(self.image)
        output += np.flipud(np.fliplr(self.image))

        self.image = output/mask2
        (l,m) = self.image.shape
        self.image = self.image[l//2:,m//2:]
        return self.image
    def ShowImage(self,axis):
        axis.imshow(np.log(self.image+1))

    def copy(self):
        return deepcopy(self)

class fiber_data():
    """
    Container to hold all relevent information about one saxs/waxs data acquisition
    """

    def __init__(self,saxs = [None,None],waxs = [None,None],beamstop_intensity = None):
        self.saxs_image = saxs[0] #np array
        self.saxs_detector = saxs[1] #see detector class
        self.waxs_image = waxs[0]
        self.waxs_detector = waxs[1]
        self.beamstop_intensity = beamstop_intensity



    def __str__(self):
        print(f"beamstop intensity: {self.beamstop_intensity}")
        if self.saxs_image is not None:
            print(f"saxs image of shape {self.saxs_image.shape} of dtype {self.saxs_image.dtype}")
            print(self.saxs_detector)

        if self.waxs_image is not None:
            print(f"waxs image of shape {self.waxs_image.shape} of dtype {self.waxs_image.dtype}")
            print("waxs detector:")
            print(self.waxs_detector)
        return super().__str__()

    def FiberSymmetry(self,image,det,align_threshold):
        new_image = fiber_image(image = image,
                                mask = det.mask,
                                centeri = det.centeri,
                                centerj = det.centerj)
        out = new_image.RotateAndApplySymmetry() #Returns only the +/+ quadrant since all four are identical. Use quadrant_unfold to recreate full image

        new_detector = detector(
                            detector_name = 'virtual',
                            distance = det.distance,
                            wavelength = det.wavelength,
                            shape = new_image.image.shape,
                            centeri = 0,
                            centerj = 0,
                            dqi = det.dqi,
                            dqj = det.dqj,
                            qi_label = 'axial',
                            qj_label = 'radial') #might have to transpose to get axes as intended 
        return out, new_detector

    def ApplyFiberSymmetry(self,saxs_align_threshold=15,waxs_align_threshold=15):
        if self.saxs_image is not None:
            self.saxs_image,self.saxs_detector = self.FiberSymmetry(self.saxs_image,self.saxs_detector,saxs_align_threshold)
        if self.waxs_image is not None:
            self.waxs_image,self.waxs_detector = self.FiberSymmetry(self.waxs_image,self.waxs_detector,saxs_align_threshold)


    def packh5(self,h5grp):
        h5grp['beamstop'] = self.beamstop_intensity
        if self.saxs_image is not None:
            saxsgrp = h5grp.create_group('saxs_data')
            saxsgrp['data'] = self.saxs_image
            saxsgrp['qi'] = self.saxs_detector.qi
            saxsgrp['qj'] = self.saxs_detector.qj
        if self.waxs_image is not None:
            waxsgrp = h5grp.create_group('waxs_data')
            waxsgrp['data'] = self.waxs_image
            waxsgrp['qi'] = self.waxs_detector.qi
            waxsgrp['qj'] = self.waxs_detector.qj

class fiber_stack(list):
    """
    List of fiber containers with slightly specialized list operations.
    Containers are assumed to be homogenous in structure.

    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.include = [True]*len(self)

    def append(self,fiber):
        #modified default list append function to update include index
        super().append(fiber)
        self.include.append(True)
        
    def pop(self,i):
        #modified default list pop function to also pop include index
        super().pop(i)
        self.include.pop(i)

    def AndInclude(self,newinclude):
        self.include = [A and B for (A,B) in zip(self.include,newinclude)]

    def OrInclude(self,newinclude):
        self.include = [A or B for (A,B) in zip(self.include,newinclude)]

    def getIncludeIndex(self):
        return [i for i,include in enumerate(self.include) if include]

    def RemoveExcluded(self):
        """
        Cleans up the stack by removing all excluded entries from stack. 
        Reminder: This is a destructive function
        """
        #Walk backwards and pop
        N = len(self.stack)
        for i in range(N-1,0,-1):
            if ~self.include[i]:
                self.pop(i)

    def ReturnSubStack(self,index):
    	return FiberStack([self.stack[i] for i in index])

    def getAttribute(self,Attribute,exclude=True):
        if exclude:
            return [getattr(Fiber,Attribute) for (include,Fiber) in zip(self.include,self) if include]
        else:
            return [getattr(Fiber,Attribute)  for Fiber in self]

    def getSubAttribute(self,Attribute,SubAttribute,exclude=True):
        if exclude:
            return [getattr(getattr(Fiber,Attribute),SubAttribute) for (include,Fiber) in zip(self.include,self) if include]
        else:
            return [getattr(getattr(Fiber,Attribute),SubAttribute)  for Fiber in self]


def quadrant_unfold(image): 
    #image is the +/+ quadrant. This function unfolds it into four quadrants
    n,m = image.shape
    new = np.zeros((2*n-1,2*m-1))
    new[:n,:m] = np.flipud(np.fliplr(image))
    new[:n,m:] = np.flipud(image[:,:-1])
    new[n:,:m] = np.fliplr(image[:-1,:])
    new[n:,m:] = image[:-1,:-1]
    return new

