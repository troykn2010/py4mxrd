import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from .background_fits import monotonic
from .FiberDiffraction import fiber_stack,fiber_data,fiber_image


cfactor = 2*3.14159/10#1/nm to 1/angstroms


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

def NGaussiansClusterError(params,x,y):
    z = -y
    s = params[-1]
    for i in range(0,len(params)-1,2):
        A = params[i]
        q = params[i+1]
        z += gauss(x,A,q,s)
    return np.sum(z**2)

class MuscleLineData():
    def __init__(self,q,y,quiet = True):
        self.q = q
        self.values = y
        self.filtered_values = None
        self.background = None
        self.quiet = quiet  
        self.peaks = {}
        self.fitted_values = None


    def FitSingleGaussian(self,x,y,label = None,bounds = None, fitmethod = 'Nelder-Mead',tol=1e-8,maxiter=1e4,quiet=True):
        """
            Presume a clean gaussian that's roughly centered.
        """
        if y.max()<1e-4:
            #Error handling for vector input of zeros.
            m0 = 0
            m1 = m0
            m2 = 0
            success = False

        amplitude = y.max()
        y = y/y.max()
        a0 = np.trapezoid(y,x) +1e-10
        a1 = np.trapezoid(y*x,x) +1e-10

        p0 = [1,a1/a0,(x.max()-x.min())/6] #[0th,1st,2nd moments]
        if bounds == None:
            bounds =[(0,1) ,(x.min(),x.max()),(1e-5,(x.max()-x.min())/2+1e-10)]

        fit = minimize(NGaussiansError,p0,args = (x,y),bounds = bounds,
        method = fitmethod ,tol = tol,options={'maxiter':maxiter})
        success = fit.success
        m0 = amplitude*fit.x[0] #unpack moments
        m1 = fit.x[1]
        m2 = fit.x[2]
        if label == None:
            label = f"{1/m1:0.3f}"
        self.peaks[label] = {}
        self.peaks[label]['m2'] = m2
        self.peaks[label]['m1'] = m1
        self.peaks[label]['m0'] = m0
        self.peaks[label]['Area'] = np.sqrt(2*np.pi)*m0*m2
        self.peaks[label]['fitsuccess'] = success
        self.peaks[label]['qmin'] = bounds[1][0]
        self.peaks[label]['qmax'] = bounds[1][1]+1e-6

        if quiet==False:
            print(fit.success)
        return self.peaks[label]

    def NGaussianFit(self,listpeaks,delta= 0.5,method = 'Nelder-Mead',tol=1e-8,maxiter=1e4):     
        #Use gaussian peaks from generated from FitSingleGaussian as an initial guess for an N-Gaussian fit.
        # Best when multiple gaussians are overlapping
        a = 1 + delta
        b = 1 - delta
        p0 = []
        bnd = []

        qmin_all = self.q.max()
        qmax_all = self.q.min()
        for peak in listpeaks:
            qmin_all = min(qmin_all,peak['qmin'])
            qmax_all = max(qmax_all,peak['qmax'])
            m0 = peak['m0']
            m1 = peak['m1']
            m2 = peak['m2']
            p0 = p0 +[m0,m1,m2+1e-10] #m2=0 generates an division by zero error

            bnd= bnd + [(m0*b,m0*a),
                        (peak['qmin'],peak['qmax']), 
                        (peak['smin'],peak['smax'])
                        ]
        #Each peak has a qmin,qmax pair that defines the sandbox limits.
        #Take the largest range that contains each peak's limits
        bool = np.logical_and(self.q>qmin_all,self.q<qmax_all)
        fit = minimize(NGaussiansError,p0,args = (self.q[bool],self.filtered_values[bool]),bounds = bnd,
         method = method ,tol = tol,options={'maxiter':maxiter})
        
        #update peaks
        j = 0
        for i in range(0,len(fit.x),3):
            listpeaks[j]['m0'] = fit.x[i]
            listpeaks[j]['m1'] = fit.x[i+1]
            listpeaks[j]['m2'] = fit.x[i+2]
            listpeaks[j]['Area'] = np.sqrt(2*np.pi)*fit.x[i]*fit.x[i+2]
            listpeaks[j]['fitsuccess'] = fit.success
            j = j+1
        return listpeaks

    def FitClusterWithGaussians(self,keys,delta= 0.5,method = 'Nelder-Mead',tol=1e-8,maxiter=1e4):     
        #Use gaussian peaks from generated from FitSingleGaussian as an initial guess for an N-Gaussian fit.
        # Best when multiple gaussians are overlapping
        listpeaks = []
        for key in keys:
            listpeaks.append(self.peaks[key])

        a = 1 + delta
        b = 1 - delta
        p0 = []
        bnd = []

        qmin_all = self.q.max()
        qmax_all = self.q.min()
        for peak in listpeaks:
            qmin_all = min(qmin_all,peak['qmin'])
            qmax_all = max(qmax_all,peak['qmax'])
            m0 = peak['m0']
            m1 = peak['m1']
            m2 = peak['m2']
            p0 = p0 +[m0,m1] #m2=0 generates an division by zero error

            bnd= bnd + [(m0*b,m0*a),
                        (peak['qmin'],peak['qmax'])
                        ]
        p0 = p0 + [m2+1e-10]
        bnd = bnd + [(peak['smin'],peak['smax'])]

        #Each peak has a qmin,qmax pair that defines the sandbox limits.
        #Take the largest range that contains each peak's limits
        bool = np.logical_and(self.q>qmin_all,self.q<qmax_all)
        fit = minimize(NGaussiansClusterError,p0,args = (self.q[bool],self.filtered_values[bool]),bounds = bnd,
         method = method ,tol = tol,options={'maxiter':maxiter})

        #update peaks
        j = 0
        for i in range(0,len(fit.x)-1,2):
            listpeaks[j]['m0'] = fit.x[i]
            listpeaks[j]['m1'] = fit.x[i+1]
            listpeaks[j]['m2'] = fit.x[-1]
            listpeaks[j]['Area'] = np.sqrt(2*np.pi)*fit.x[i]*fit.x[-1]
            listpeaks[j]['fitsuccess'] = fit.success
            j = j+1

        for i,key in enumerate(keys):
            self.peaks[key]= listpeaks[i]

    def NGaussianFitKeys(self,keys,**kwargs):
        #Wrapper around NGaussianFit to take in keys as argument
        listpeaks = []
        for key in keys:
            listpeaks.append(self.peaks[key])
        newlistpeaks = self.NGaussianFit(listpeaks,**kwargs)
        for i,key in enumerate(keys):
            self.peaks[key]= newlistpeaks[i]

    def Peak_Data(self,peak):
        if peak['m2']>1e-6:
            return gauss(self.q, peak['m0'], peak['m1'], peak['m2'])
        else:
            return 0*self.q

    def BackgroundRemoval(self,interpolator):
        # self.backgroundinterpolator = deepcopy(interpolator)
        self.background = interpolator(self.q)
        self.filtered_values = self.values - self.background

    def ComputeFittedValues(self,keys):
        self.fitted_values = 0
        for key in keys:
            self.fitted_values += self.Peak_Data(self.peaks[key])

    def packh5(self,h5grp):
        h5grp['values'] = self.values
        h5grp['backgrounds'] = self.background
        h5grp['signals'] = self.filtered_values
        h5grp['fits'] = self.fitted_values
        h5grp['q(angstrom-1)'] = self.q
        h5grp['d(nm)'] = cfactor/self.q
        
        a = np.trapezoid(self.filtered_values*self.q/cfactor,self.q/cfactor)
        b = np.trapezoid(self.filtered_values,self.q/cfactor)
        h5grp['TotalArea(count nm-1)'] = b
        h5grp['d_COM(nm)'] = b/a

        peaks_grp = h5grp.create_group('peaks')
        for key in self.peaks.keys():
            peak_grp = peaks_grp.create_group(key)
            peak_grp['dspacing(nm)'] = cfactor/self.peaks[key]['m1']
            peak_grp['sigma(nm-1)'] = self.peaks[key]['m2']/cfactor
            peak_grp['area(count nm-1)'] = self.peaks[key]['Area']/cfactor

    def copy(self):
        return deepcopy(self)



class MuscleAreaData():
    """
    Rectilinear grid
    Everything is built on nd arrays
    """
    def __init__(self,values,qi=None,qi_label='i',qj=None,qj_label='j', detector = None,quiet = True):
        if detector is None:
            self.qi = qi
            self.qi_label = qi_label #Ask user to explicitly state coordinate directions. x,y,i,j,radial,axial
            self.qj = qj
            self.qj_label = qj_label #Ask user to explicitly state coordinate directions. x,y,i,j,radial,axial
        else:
            self.qi = detector.qi
            self.qi_label = detector.qi_label
            self.qj = detector.qj
            self.qj_label= detector.qj_label

        self.values = values
        self.filtered_values = None
        self.background = None
        self.quiet = quiet  

    def ROI(self,qi_range = [-1e10,1e10],qj_range = [-1e10,1e10]):
        qi_min = max(qi_range[0],self.qi.min())
        qi_max = min(qi_range[1],self.qi.max())
        qj_min = max(qj_range[0],self.qj.min())
        qj_max = min(qj_range[1],self.qj.max())

        booli = np.logical_and(self.qi>=qi_min,self.qi<qi_max)
        boolj = np.logical_and(self.qj>=qj_min,self.qj<qj_max)


        values = self.values[booli,:]
        values = values[:,boolj]


        return MuscleAreaData(qi_label = self.qi_label,
                              qi = self.qi[booli],
                              qj_label =self.qj_label,
                              qj = self.qj[boolj],
                              values = values,
                              quiet = self.quiet)

    def Reduce2LineData(self,reduce_direction):
        if reduce_direction == self.qi_label:
            q = self.qj
            axis = 0
        elif reduce_direction == self.qj_label:
            q = self.qi
            axis = 1
        else:
            raise Exception(f"reduce_direction needs to be either {self.qi_label} or {self.qj_label}")
        y = np.mean(self.values,axis = axis)
        y = np.squeeze(y) #drop empty dimensions
        LineData = MuscleLineData(q,y)
        return LineData

    def SubtractBackground_Monotonic_ConvexHull(self,direction):
        if direction == self.qi_label:
            values = self.values.T
            q = self.qi
        elif direction == self.qj_label:
            values = self.values
            q = self.qj
        else:
            raise Exception(f"direction needs to be either {self.qi_label} or {self.qj_label}")

        background = np.zeros(values.shape)
        filtered_values = np.zeros(values.shape)
        for i in range(len(values)):
            line = values[i]
            h = monotonic(q,line)
            background[i] = h(q)
            filtered_values[i] = line-background[i]

        if direction == self.qi_label:
            self.filtered_values = filtered_values.T
            self.background = background.T
        elif direction == self.qj_label:
            self.filtered_values = filtered_values
            self.background = background


    def proc_box(self,box):
        """
            example box:

            #Equators
            e_principalSpacing = 38 #nm
            e_peaks = {}
            e_peaks['10'] = {'relative_qmin':e_principalSpacing/44, #Max
                             'relative_qmax':e_principalSpacing/32, #Min
                             'absolute_smin':1e-4 *c,
                             'absolute_smax':4e-3 *c}
            e_peaks['11'] = {'relative_qmin':1.73*0.75, 
                             'relative_qmax':1.73*1.25,
                             'absolute_smin':1e-4 *c,
                             'absolute_smax':5e-3 *c}

            equator_box = {
                'label':'equator',
                'background_direction': 'radial',
                'reduce_direction': 'axial',
                'radial': [1/80 * c,1/15*c],
                'axial': [0,0.005 * c],
                'PrincipalSpacing': e_principalSpacing, #nm
                'peaks':e_peaks,
                'update_keys':[ ['10' ,'11']],
                'update_method': 'NGaussian'
            }

        """
        d0 = box['PrincipalSpacing']
        qi = (1/d0)*cfactor

        boxAreaData = self.ROI(qi_range=box[self.qi_label],
                               qj_range=box[self.qj_label],)

        if 'radial' in box['label']:
            #Hacky
            boxAreaData.SubtractBackground_Monotonic_ConvexHull(direction=box['background_direction'])
            boxAreaData.values = boxAreaData.filtered_values
            LineData = boxAreaData.Reduce2LineData(reduce_direction = box['reduce_direction'])
            #Background subtract already happened. This just sets filtered values to values and background to zero
            LineData.filtered_values = LineData.values
            LineData.background = np.zeros_like(LineData.q)
        else:
            LineData = boxAreaData.Reduce2LineData(reduce_direction = box['reduce_direction'])
            LineData.BackgroundRemoval(monotonic(LineData.q,LineData.values))
        for key in box['peaks'].keys():
            peak = box['peaks'][key]
            bounds = [(0,1),(qi*peak['relative_qmin'],qi*peak['relative_qmax']),(peak['absolute_smin'],peak['absolute_smax']) ] #bounds on single gaussian fit
            bool = np.logical_and(LineData.q>=qi*peak['relative_qmin'],LineData.q<=qi*peak['relative_qmax'])
            LineData.FitSingleGaussian(LineData.q[bool],LineData.filtered_values[bool],label = key,maxiter = 1000,bounds = bounds) #initial fits
            LineData.peaks[key]['smin'] = peak['absolute_smin']
            LineData.peaks[key]['smax'] = peak['absolute_smax']

        for update_keys in box['update_keys']:
            if box['update_method'] == 'NGaussian':
                LineData.NGaussianFitKeys(update_keys,maxiter=1000,delta = 0.5) 
            elif box['update_method'] == 'NGaussianCluster':
                LineData.FitClusterWithGaussians(update_keys,maxiter=1000)
            else:
                print('Update method not recognized')
        
        LineData.ComputeFittedValues(box['peaks'].keys())
        return LineData

    def copy(self):
        return deepcopy(self)

class MuscleStack(fiber_stack):
    """
    Child class of fiber_stack (which itself is a child of the list class) with specialized functions for muscles
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def proc_equators(self,equator_box,qi_label = 'axial',qj_label='radial'):
        #Compute equatorial fits on entire stack
        for fiber in self:
            AreaData = MuscleAreaData(values = fiber.saxs_image,detector = fiber.saxs_detector)
            fiber.equator = AreaData.proc_box(equator_box)

        self.d10 = [cfactor/fiber.equator.peaks['10']['m1'] for fiber in self]
        self.IR = [fiber.equator.peaks['11']['Area']/fiber.equator.peaks['10']['Area'] for fiber in self]
        self.Area10 = [fiber.equator.peaks['10']['Area'] for fiber in self]

    def merge(self,d10min=34,d10max=42,IRmin = 0 , IRmax = 1,Area10min = 0,Area10max = 10,remove = True):
        if remove:
            self.AndInclude([ d10>d10min for d10 in self.d10])
            self.AndInclude([ d10<d10max for d10 in self.d10])
            self.AndInclude([ IR>IRmin for IR in self.IR])
            self.AndInclude([ IR<IRmax for IR in self.IR])
            self.AndInclude([ Area10>Area10min for Area10 in self.Area10])
            self.AndInclude([ Area10<Area10max for Area10 in self.Area10])

        if self[0].saxs_image is not None:
            saxs_merged = np.mean(self.getAttribute('saxs_image',exclude = True),axis = 0)
        else:
            saxs_merged = None
        if self[0].waxs_image is not None:
            waxs_merged = np.mean(self.getAttribute('waxs_image',exclude = True),axis = 0)
        else:
            waxs_merged = None
        beamstop_intensity = np.mean(self.getAttribute('beamstop_intensity',exclude=True))

        self.merged = fiber_data(saxs = [saxs_merged,self[0].saxs_detector],
                          waxs = [waxs_merged,self[0].waxs_detector],
                          beamstop_intensity = beamstop_intensity)
        return self.merged,int(sum(self.include))

    def packh5(self,h5grp,exclude = False):
        #packs stack data into h5 file
        equatorgrp = h5grp.create_group('equator')
        equatorgrp['signals'] = self.getSubAttribute(Attribute='equator',SubAttribute='filtered_values',exclude=exclude)
        equatorgrp['values'] = self.getSubAttribute(Attribute='equator',SubAttribute='values',exclude=exclude)
        equatorgrp['backgrounds'] = self.getSubAttribute(Attribute='equator',SubAttribute='background',exclude=exclude)
        equatorgrp['fits'] = self.getSubAttribute(Attribute='equator',SubAttribute='fitted_values',exclude=exclude)
        equatorgrp['q(angstrom-1)'] = self[0].equator.q
        equatorgrp['d(nm)'] = cfactor/self[0].equator.q

        peaks = self.getSubAttribute(Attribute='equator',SubAttribute='peaks',exclude=exclude)
        equatorgrp['peaks/10/dspacing(nm)']     = self.d10
        equatorgrp['peaks/10/area(count nm-1)'] = [peak['10']['Area']/cfactor for peak in peaks]
        equatorgrp['peaks/10/sigma(nm-1)']      = [peak['10']['m2']/cfactor for peak in peaks]

        equatorgrp['peaks/11/dspacing(nm)']     = [cfactor/peak['11']['m1'] for peak in peaks]
        equatorgrp['peaks/11/area(count nm-1)'] = [peak['11']['Area']/cfactor for peak in peaks]
        equatorgrp['peaks/11/sigma(nm-1)']      = [peak['11']['m2']/cfactor for peak in peaks]

        equatorgrp['IR'] = self.IR


        h5grp['beamstop'] = self.getAttribute(Attribute='beamstop_intensity',exclude=exclude)

        if self[0].saxs_image is not None:
            saxsgrp = h5grp.create_group('saxs_data')
            saxsgrp['data'] = self.getAttribute(Attribute='saxs_image',exclude=exclude)
            saxsgrp['qi'] = self[0].saxs_detector.qi
            saxsgrp['qj'] = self[0].saxs_detector.qj
        if self[0].waxs_image is not None:
            waxsgrp = h5grp.create_group('waxs_data')
            waxsgrp['data'] = self.getAttribute(Attribute='waxs_image',exclude=exclude)
            waxsgrp['qi'] = self[0].waxs_detector.qi
            waxsgrp['qj'] = self[0].waxs_detector.qj
