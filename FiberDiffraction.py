import numpy as np
import cv2
from scipy.optimize import minimize
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
        self.equator = None
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
    def CenterByPadding(image,centeri,centerj):
        #numpy ij indexing
        l = image.shape[0]
        m = image.shape[1]
        #Move to center
        if 2*centeri<=l:
            pad0 = (int(l-2*centeri),0)
        else:
            pad0 = (0,int(2*centeri-l))

        if 2*centerj<=m:
            pad1 = (int(m-2*centerj),0)
        else:
            pad1 = (0,int(2*centerj-m))
        image = np.pad(image, (pad0,pad1)) 

        #Pad again to avoid information lost when rotating
        # R = int(np.ceil(np.sqrt((l//2)**2 + (m//2)**2)))
        # image = np.pad(image, (  (R-l//2,R-l//2),(R-m//2,R-m//2) )  )
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
        
    def RotateAndApplySymmetry(self,align='auto'):
        self.image = self.CenterByPadding(self.image*self.mask,self.centeri,self.centerj)
        self.mask = self.CenterByPadding(self.mask,self.centeri,self.centerj)

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
        
    def ShowImage(self,axis):
        axis.imshow(np.log(self.image+1))

def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))    

def NGaussiansError(params,x,y):
    z = -y
    for i in range(0,len(params),3):
        A = params[i]
        q = params[i+1]
        s = params[i+2]
        z += gauss(x,A,q,s)
    return np.sum(z**2)

class MuscleMeridonials():
    """
    Let's not add to the zoo of inconsistent meridonial nomenclature.
    All reflections will be labeled by the spacing ~i.e 42.9nm, 14.3nm and so on.
    """

    def __init__(self,q,y,quiet=True):
        self.q = q #non-cartesian grid due to combination of saxs/waxs
        self.values = y
        self.filtered_values = None
        self.background = None
        self.quiet = quiet

    def FindPrincipalMyosinSpacing(method ='WAXS'):
        """
         Autonomously find first order myosin spacing.
         Should be around 42.9nm.
        """

        if method == 'WAXS':
            """
                Uses higher order reflection in waxs detector to find spacing.
                Best use case is in-air acquisition. 
            """
            None


        return f"Method {method} not recognized"



    def FindPrincipalActinSpacing():
        """
         Autonomously find first order actin spacing.
         Should be around 2.7nm.
        """

def BackgroundRemoval(x,y,interpolator):
    # Leave it to the user to decide which background to use
    background = interpolator(x)
    filtered_values = y - background
    return background,filtered_values

class MuscleEquator():
    def __init__(self,q,y,quiet = True,zdisc=False):
        #q is a 1d coordinate array in q-spare
        #equator is a 1d intensity array
        self.q = q
        self.values = y
        self.filtered_values = None
        self.background = None
        self.quiet = quiet
        self.zdisc = zdisc

        #Initialize fit values
        self.ResetFit()

        # self.I10     = np.sqrt(2*np.pi)*self.A10*self.s10
        # self.I11     = np.sqrt(2*np.pi)*self.A11*self.s11
        # self.IR      = self.A11*self.s11/(self.A10+1e-9)/(self.s10+1e-9)

    def NGaussianFit(self,guess,bounds,method = 'Nelder-Mead',tol=1e-8,maxiter=1e4):
        #Guess is a dictionary with initial values on q10,A10,s10,q11,A11,s11 and optionally q_zdisc, A_zdisc, and s_zdisc. 
        #Bounds is a dictionary with 2-tuple (low,high) instances on q10,A10,s10,q11,A11,s11 and optionally q_zdisc, A_zdisc, and s_zdisc. 
        
        #unpack into list for scipy minimize
        if self.zdisc:
            p0 = [guess['A10']    ,guess['q10']    ,guess['s10'],
                  guess['A11']    ,guess['q11']    ,guess['s11'],
                  guess['A_zdisc'],guess['q_zdisc'],guess['s_zdisc']]
            bnd =[bounds['A10']    ,bounds['q10']    ,bounds['s10'],
                  bounds['A11']    ,bounds['q11']    ,bounds['s11'],
                  bounds['A_zdisc'],bounds['q_zdisc'],bounds['s_zdisc']]
        else:
            p0 = [guess['A10']    ,guess['q10']    ,guess['s10'],
                  guess['A11']    ,guess['q11']    ,guess['s11']]
            bnd =[bounds['A10']    ,bounds['q10']    ,bounds['s10'],
                  bounds['A11']    ,bounds['q11']    ,bounds['s11']]           
        fit = minimize(NGaussiansError,p0,args = (self.q,self.filtered_values),bounds = bnd,
         method = method ,tol = tol,options={'maxiter':maxiter})

        #update fit
        self.fitsuccess = fit.success
        #10 Gaussian values
        self.fit['A10'] = fit.x[0] #Amplitude
        self.fit['q10'] = fit.x[1] #mean
        self.fit['s10'] = fit.x[2] #standard deviation
        self.fit['y10'] = gauss(self.q,fit.x[0],fit.x[1],fit.x[2]) #Gaussian trace
        
        #11 Gaussian values
        self.fit['A11'] = fit.x[3] 
        self.fit['q11'] = fit.x[4]
        self.fit['s11'] = fit.x[5]
        self.fit['y11'] = gauss(self.q,fit.x[3],fit.x[4],fit.x[5])  

        #Integrated values
        self.fit['I10'] = np.sqrt(2*np.pi)*self.fit['A10']*self.fit['s10']
        self.fit['I11'] = np.sqrt(2*np.pi)*self.fit['A11']*self.fit['s11']
        self.fit['IR']  = self.fit['A11']*self.fit['s11']/(self.fit['A10']+1e-9)/(self.fit['s10']+1e-9)
        
        self.fit['y']   = self.fit['y10'] + self.fit['y11']
        if self.zdisc:
            #z-disc Gaussian values
            self.fit['A_zdisc'] = fit.x[6]
            self.fit['q_zdisc'] = fit.x[7]
            self.fit['s_zdisc'] = fit.x[8]
            self.fit['y_zdisc'] = gauss(self.q,fit.x[6],fit.x[7],fit.x[8]) #Gaussian trace
            self.fit['y']   += self.fit['y_zdisc']
                                            
    def ResetFit(self):
        self.fit = {}
        #10 Gaussian values
        self.fit['A10'] = 1e-8 #Amplitude 
        self.fit['q10'] = 1e8 #mean
        self.fit['s10'] = 1e-8 #standard deviation
        self.fit['y10'] = [0]*len(self.q) #Gaussian trace
        
        #11 Gaussian values
        self.fit['A11'] = 1e-8 
        self.fit['q11'] = 1e8
        self.fit['s11'] = 1e-8
        self.fit['y11'] = 1e-8 

        #z-disc Gaussian values
        self.fit['A_zdisc'] = 1e-8
        self.fit['q_zdisc'] = 1e8
        self.fit['s_zdisc'] = 1e-8
        self.fit['y_zdisc'] = [0]*len(self.q) #Gaussian trace

        #Integrated values
        self.fit['I10'] = 1e-8
        self.fit['I11'] = 1e-8
        self.fit['IR']  = 1e-8
        self.fit['y']   = [0]*len(self.q)   
    
    #Upgraded from class function to stand-alone static function
    # def BackgroundRemoval(self,interpolator):
    #     # self.backgroundinterpolator = deepcopy(interpolator)
    #     self.background = interpolator(self.q)
    #     self.filtered_values = self.values - self.background

    def ShowRawPlot(self,axis):
        axis.plot(self.q,self.values)
        if self.background is not None:
            axis.plot(self.q,self.background,color = 'k',zorder = -1)
        
    def ShowRawLogLog(self,axis):
        axis.loglog(self.q,self.values)
        if self.background is not None:
            axis.loglog(self.q,self.background)
        
    def ShowFilteredPlot(self,axis):
        axis.plot(self.q,self.filtered_values)
        if self.fit['y'] is not None:
            axis.plot(self.q,self.fit['y10'])
            axis.plot(self.q,self.fit['y11'])
            axis.plot(self.q,self.fit['y'])
            if self.zdisc:
                axis.plot(self.q,self.fit['y_zdisc'])
        
    def ShowFilteredLogLog(self,axis):
        axis.loglog(self.q,self.filtered_values)
        if self.fit['y'] is not None:
            axis.loglog(self.q,self.fit['y10'])
            axis.loglog(self.q,self.fit['y11'])
            axis.loglog(self.q,self.fit['y'])
            if self.zdisc:
                axis.loglog(self.q,self.fit['y_zdisc'])
        

class MuscleStack():
    """
    List of fiber_image objects
    """
    def __init__(self,stack):
        self.stack = stack
        self.include = [True]*len(stack)

    def append(self,myosaxs):
        """
        Append new myosaxs object
        """
        self.stack.append(myosaxs)
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
        N = len(stack)
        for i in range(N,0,-1):
            if ~self.include[i]:
                self.stack.pop(i)
                self.include.pop(i)
                
    def getImages(self,exclude = True):
        """
        Returns stack of fiber images as a list
        optional input exclude = True ignores all entries that have been excluded
        """
        if exclude:
            return [myosaxs.image for (include,myosaxs) in zip(self.include,self.stack) if include]
        else:
            return [myosaxs.image for myosaxs in self.stack]

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
            
        