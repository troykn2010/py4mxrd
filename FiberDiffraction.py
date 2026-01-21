import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt

class fiber_image():
    def __init__(self,image,mask,centeri=0,centerj=0,align_threshold=15,AutoCentering=False,quiet=True,dq = None,phi = 0):
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

        #Pad again to avoid information lost when rotating
        # R = int(np.ceil(np.sqrt((l//2)**2 + (m//2)**2)))
        # image = np.pad(image, (  (R-l//2,R-l//2),(R-m//2,R-m//2) )  )

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
                [1, 0, -ishift],
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
        # self.output = output
        # self.mask2 = mask2
    def ShowImage(self,axis):
        axis.imshow(np.log(self.image+1))

class Fiber():
	"""
	Currently trivial container for various fiber objects.
	To expand on functionality as need arises
	"""
	saxs = None
	waxs = None
	meridonials = None
	equators = None


class FiberStack():
    """
    List of fiber containers with slightly specialized list operations.
    Containers are assumed to be homogenous in structure
    """
    def __init__(self,stack):
        self.stack = stack
        self.include = [True]*len(stack)

    def append(self,Fiber):
        """
        Append new Fiber container
        """
        self.stack.append(Fiber)
        self.include.append(True)
        
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
                self.stack.pop(i)
                self.include.pop(i)

    def ReturnSubStack(self,index):
    	return FiberStack([self.stack[i] for i in index])
                
    # def getImages(self,exclude = True):
    #     """
    #     Returns stack of fiber images as a list
    #     optional input exclude = True ignores all entries that have been excluded
    #     """
    #     if exclude:
    #         return [myosaxs.image for (include,myosaxs) in zip(self.include,self.stack) if include]
    #     else:
    #         return [myosaxs.image for myosaxs in self.stack]
    def getAttribute(self,Attribute,exclude=True):
        if exclude:
            return [getattr(Fiber,Attribute) for (include,Fiber) in zip(self.include,self.stack) if include]
        else:
            return [getattr(Fiber,Attribute)  for Fiber in self.stack]

    def getSubAttribute(self,Attribute,SubAttribute,exclude=True):
        if exclude:
            return [getattr(getattr(Fiber,Attribute),SubAttribute) for (include,Fiber) in zip(self.include,self.stack) if include]
        else:
            return [getattr(getattr(Fiber,Attribute),SubAttribute)  for Fiber in self.stack]

    def getEquatorAttribute(self,Attribute,exclude = True):
        """
        Returns equator attributes as a list. Supported attributes are: 

        optional input exclude = True ignores all entries that have been excluded
        
        q10: q-space position of equatorial 1,0 reflection
        A10: amplitude of equatorial 1,0 reflection
        s10: sigma of gaussian fit in q-space of equatorial 1,0 reflection
        I10: Area under the gaussian fit of 1,0 reflection
    
        q11: q-space position of equatorial 1,1 reflection
        Aratio: A11/A10 where A11 is the amplitude of equatorial 1,1 reflection
        s11: sigma of gaussian fit in q-space of equatorial 1,1 reflection
        I11: Area under the gaussian fit of 1,1 reflection
    
        IR: I11/I10
    
        q: q-values
        fitted_10: trace for gaussian fit of 10
        fitted_11: trace for gaussian fit of 11
        fitted_values: trace of total fit
        """
        if exclude:
            return [getattr(myosaxs.equator,Attribute) for (include,myosaxs) in zip(self.include,self.stack) if include]
        else:
            return [getattr(myosaxs.equator,Attribute) for myosaxs in self.stack]
            

        
    # def Export2CSV(self, filename ,ListMuscleLineDataName):
    #     #First, figure out all columns
    #     fiber = self.stack[0]
    #     ListPeaks = []
    #     for MuscleLineDataName in ListMuscleLineDataName:
    #         peaks = getattr(getattr(fiber,MuscleLineDataName),'peaks')
    #         for key in peaks.keys():
    #             ListPeaks.append([MuscleLineDataName,key])

    #     for fiber in self.stack:
    #         for s in ListPeaks:
    #             peak = getattr(fiber,s[0][0]).peaks[s[0][1]]

