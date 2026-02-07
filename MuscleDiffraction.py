import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

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
        self.peaks[label]['qmin'] = x.min()
        self.peaks[label]['qmax'] = x.max()+1e-6

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

class MuscleMeridionals(MuscleLineData):
    """
    Let's not add to the zoo of inconsistent meridional nomenclature.
    All reflections will be labeled by the spacing ~i.e 42.9nm, 14.3nm and so on.
    """
    def __init__(self,q,y,quiet = True):
        super().__init__(q=q,y=y,quiet=quiet)
        self.PrincipleMyosinSpacing = None
        self.PrincipleMyosinSpacing = None

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
        fit = minimize(self.NGaussiansClusterError,p0,args = (self.q[bool],self.filtered_values[bool]),bounds = bnd,
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

    def NGaussiansClusterError(self,params,x,y):
        z = -y
        s = params[-1]
        for i in range(0,len(params)-1,2):
            A = params[i]
            q = params[i+1]
            z += gauss(x,A,q,s)
        return np.sum(z**2)

    def FindPrincipleActinSpacing(self,method = '13thOrder',quiet = True):
        """
         Autonomously find gobular actin on the 13th order
         Should be around 2.7nm.
        """
        bool = np.logical_and(self.q>1/2.8,self.q<1/2.6) 
        q = self.q[bool]
        y = self.filtered_values[bool]
        peak = self.FitSingleGaussian(q,y,quiet=quiet)

        self.PrincipleActinSpacing = 13/peak['m1']
        return peak

    def FindPrincipleMyosinSpacing(self,method ='3rdOrder',quiet = True):
        """
         Autonomously find first order myosin spacing.
         Should be around 42.9nm.
        """

        if method == 'WAXS':
            """
                Uses higher order reflection in waxs detector to find spacing.
                Best use case is in-air acquisition. 
            """

            #In development
        elif method == '3rdOrder':
            """
                Uses the peak near the third reflection to estimate spacing
            """
            bool = np.logical_and(self.q>1/15.5,self.q<1/13.5) #Assume the third order refleciton is between 13.5nm and 15.5nm
            q = self.q[bool]
            y = self.filtered_values[bool]
            peak = self.FitSingleGaussian(q,y,label ='AutoM3',quiet=quiet)

            self.PrincipleMyosinSpacing = 3/peak['m1']

            return peak
        return f"Method {method} not recognized"

    def PeakbyCentroid(self,x,y,label = None):
        a = np.trapz(x*y,x)
        b = np.trapz(y,x)

        x0 = a/b
        if label == None:
            label = f"{1/x:0.3f}"
        self.peaks[label] = {}
        self.peaks[label]['m1'] = x0
        self.peaks[label]['m2'] = 0
        self.peaks[label]['m0'] = y.max()
        self.peaks[label]['Area'] = b
        return self.peaks[label]


class MuscleEquator(MuscleLineData):
    def __init__(self,q,y,quiet = True):
        super().__init__(q=q,y=y,quiet=quiet)
        self.d10 = None
        self.d11 = None
        self.IR = None
        # self.I10     = np.sqrt(2*np.pi)*self.A10*self.s10
        # self.I11     = np.sqrt(2*np.pi)*self.A11*self.s11
    def Find_10(self,dmin = 32,dmax = 44,quiet = True,bounds = None):
        bool = np.logical_and(self.q>1/dmax,self.q<1/dmin)
        q = self.q[bool]
        y = self.filtered_values[bool]        
        peak = self.FitSingleGaussian(q,y,label = '10',bounds = bounds)
        self.d10 = 1/peak['m1']

    #Depreciated. Use NGaussianFitKeys
    def EquatorFit(self,keys = ['10','11'],delta = 0.5,quiet = True):
        listpeaks = []
        for key in keys:
            listpeaks.append(self.peaks[key])
        newlistpeaks = self.NGaussianFit(listpeaks,delta = delta)
        for i,key in enumerate(keys):
            self.peaks[key]= newlistpeaks[i]


    def ComputeIR(self):
        self.d11 = 1/self.peaks['11']['m1']
        self.IR = self.peaks['11']['Area']/(self.peaks['10']['Area']+1e-9)
        self.fitsuccess = np.all([self.peaks[key]['fitsuccess'] for key in self.peaks.keys()])
