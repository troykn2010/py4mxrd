import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from .background_fits import monotonic

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
        a0 = np.trapz(y,x) +1e-10
        a1 = np.trapz(y*x,x) +1e-10

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

    def copy(self):
        return deepcopy(self)


class MuscleAreaData():
    """
    Rectilinear grid
    Everything is built on nd arrays
    """
    def __init__(self,q0_label,q0,q1_label,q1,values,quiet = True):
        self.q0 = q0
        self.q0_label = q0_label #Ask user to explicitly state coordinate directions. x,y,i,j,radial,azimuthal
        self.q1 = q1
        self.q1_label = q1_label #Ask user to explicitly state coordinate directions. x,y,i,j,radial,azimuthal

        self.values = values
        self.filtered_values = None
        self.background = None
        self.quiet = quiet  

    def ROI(self,q0_range = [-1e10,1e10],q1_range = [-1e10]):
        q0_min = max(q0_range[0],self.q0.min())
        q0_max = min(q0_range[1],self.q0.max())
        q1_min = max(q1_range[0],self.q1.min())
        q1_max = min(q1_range[1],self.q1.max())

        bool0 = np.logical_and(self.q0>=q0_min,self.q0<q0_max)
        bool1 = np.logical_and(self.q1>=q1_min,self.q1<q1_max)


        values = self.values[bool0,:]
        values = values[:,bool1]
        return MuscleAreaData(q0_label = self.q0_label,
                              q0 = self.q0[bool0],
                              q1_label =self.q1_label,
                              q1 = self.q1[bool1],
                              values = values,
                              quiet = self.quiet)

    def Reduce2LineData(self,reduce_direction):
        if reduce_direction == self.q0_label:
            q = self.q1
            axis = 0
        elif reduce_direction == self.q1_label:
            q = self.q0
            axis = 1
        else:
            raise Exception(f"reduce_direction needs to be either {self.q0_label} or {self.q1_label}")
        y = np.mean(self.values,axis = axis)
        y = np.squeeze(y) #drop empty dimensions
        LineData = MuscleLineData(q,y)
        return LineData

    def SubtractBackground_Monotonic_ConvexHull(self,direction):
        if direction == self.q0_label:
            values = self.values.T
            q = self.q0
        elif direction == self.q1_label:
            values = self.values
            q = self.q1
        else:
            raise Exception(f"direction needs to be either {self.q0_label} or {self.q1_label}")

        background = np.zeros(values.shape)
        filtered_values = np.zeros(values.shape)
        for i in range(len(values)):
            line = values[i]
            h = monotonic(q,line)
            background[i] = h(q)
            filtered_values[i] = line-background[i]

        if direction == self.q0_label:
            self.filtered_values = filtered_values.T
            self.background = background.T
        elif direction == self.q1_label:
            self.filtered_values = filtered_values
            self.background = background


    def proc_box(self,box):
        """
            example box:

            c = 2*3.14159/10#1/nm to 1/angstroms
            equator_box = {
                'label':'equator',
                'background_direction': 'x',
                'reduce_direction': 'y',
                'q0_min': 1/80 * c,
                'q0_max': 1/15  * c,
                'q1_min': 0,
                'q1_max': 0.005 * c,
                'PrincipalSpacing':38, #nm
                'peaks':e_peaks,
                'update_keys':[ ['10' ,'11']],
                'update_method': 'NGaussian'
            }
        """
        d0 = box['PrincipalSpacing']
        q0 = (1/d0)*cfactor

        boxAreaData = self.ROI(q0_range=[box['q0_min'],box['q0_max']],
                               q1_range=[box['q1_min'],box['q1_max']],)

        if 'radial' in box['label']:
            #Hacky
            boxAreaData.SubtractBackground_Monotonic_ConvexHull(direction=box['background_direction'])
            boxAreaData.values = boxAreaData.filtered_values
            LineData = boxAreaData.Reduce2LineData(reduce_direction = box['reduce_direction'])
            #Background subtract already happened. This just sets filtered values to values and background to zero
            LineData.filtered_values = LineData.values
            LineData.background = 0*LineData.q
        else:
            LineData = boxAreaData.Reduce2LineData(reduce_direction = box['reduce_direction'])
            LineData.BackgroundRemoval(monotonic(LineData.q,LineData.values))
        for key in box['peaks'].keys():
            peak = box['peaks'][key]
            bounds = [(0,1),(q0*peak['relative_qmin'],q0*peak['relative_qmax']),(peak['absolute_smin'],peak['absolute_smax']) ] #bounds on single gaussian fit
            bool = np.logical_and(LineData.q>=q0*peak['relative_qmin'],LineData.q<=q0*peak['relative_qmax'])
            LineData.FitSingleGaussian(LineData.q[bool],LineData.filtered_values[bool],label = key,maxiter = 1000,bounds = bounds) #initial fits
            LineData.peaks[key]['smin'] = peak['absolute_smin']
            LineData.peaks[key]['smax'] = peak['absolute_smax']

        for update_keys in box['update_keys']:
            if box['update_method'] == 'NGaussian':
                LineData.NGaussianFitKeys(update_keys,maxiter=1000,delta = 0.5) #Fit 10 and 11 together using initial fits as guesses
            elif box['update_method'] == 'NGaussianCluster':
                LineData.FitClusterWithGaussians(update_keys,maxiter=1000)
            else:
                print('Update method not recognized')
        
        LineData.ComputeFittedValues(box['peaks'].keys())
        return LineData

    def copy(self):
        return deepcopy(self)