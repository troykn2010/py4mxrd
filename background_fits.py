import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def BackgroundRemoval(x,y,interpolator):
    # Leave it to the user to decide which background to use
    background = interpolator(x)
    filtered_values = y - background
    return background,filtered_values

#this one works well for cardiac samples
def polynomial_background(equator,correction_factor = None):
    #Background removal
    ptslow = np.array([[x,y] for x,y in zip(equator.q,equator.values) if x <1/55])
    p = np.polyfit(np.log(ptslow[:,0]),np.log(ptslow[:,1]),1)

    ptshigh = np.array([[x,y] for x,y in zip(equator.q,equator.values) if x >1/20])
    w = np.polyfit(np.log(ptshigh[:,0]),np.log(ptshigh[:,1]),1)
    
    if correction_factor is None:
        correction_factor = 1
      
    def h(x):
        low = np.exp(p[1])*x**p[0]
        high = np.exp(w[1])*x**w[0]
        midpoint = 1/30
        a = 1/(1+np.exp(-(x-midpoint)/0.005)) #sigmoidal transition between two polynomials
        return (low*(1-a) + high*a)*correction_factor
    return h


def monotonic(q,y):
    z = y
    q = np.append(q,[0.99*q.min(),1.01*q.max()])
    z = np.append(z,[2*z.max()+1e-12,2*z.max()+1e-12])
    points = np.array([q,z]).transpose()
    hull = ConvexHull(points)
    hullpoints = np.array([[points[vertex, 0], points[vertex, 1]] for vertex in hull.vertices ])
    h = interp1d(hullpoints[:,0],hullpoints[:,1],bounds_error=False)
    return h


def loglog_convexhull(equator,correction_factor = None):
    q = equator.q
    y = equator.values
    z = interp1d(q,y)

    low = np.logical_and(q < 0.018,q>0.01)
    mid = np.logical_and(q>0.31,q<0.033)
    high = q> 0.05
    bool = np.logical_or(low,mid)
    bool = np.logical_or(bool,high)

    points = np.array([q[bool],y[bool]]).transpose()
    hull = ConvexHull(points)
    hullpoints = np.array([[points[vertex, 0], points[vertex, 1]] for vertex in hull.vertices ])
    g = interp1d(np.log(hullpoints[:,0]),np.log(hullpoints[:,1]),bounds_error=False)

    newq = np.linspace(np.log(q.min()),np.log(q.max()),100)
    newy = gaussian_filter1d(g(newq),5,mode = 'nearest')
    newg = interp1d(newq,newy)
    
    if correction_factor is None:
        correction_factor = 1
        # correction_factor = np.log(z(0.015))/g(np.log(0.015))
    def h(x):
        return np.exp(newg(np.log(x))*correction_factor)
    return h

# def loglog_convexhull(equator,correction_factor = None):
#     q = equator.q
#     y = equator.values
#     z = interp1d(q,y)

#     low = np.logical_and(q < 0.018,q>0.01)
#     mid = np.logical_and(q>0.031,q<0.033)
#     high = q> 0.05
#     bool = np.logical_or(low,mid)
#     bool = np.logical_or(bool,high)

#     points = np.array([np.log(q[bool]),np.log(y[bool])]).transpose()
#     hull = ConvexHull(points)
#     hullpoints = np.array([[points[vertex, 0], points[vertex, 1]] for vertex in hull.vertices ])
#     g = interp1d(hullpoints[:,0],hullpoints[:,1],bounds_error=False)



#     if correction_factor is None:
#         correction_factor = np.log(z(0.015))/g(np.log(0.015))
#     def h(x):
#         return np.exp(g(np.log(x))*correction_factor)
#     return h


def loglog_gaussian(equator,correction_factor = None):
    q = equator.q
    y = equator.values
    z = interp1d(q,y,bounds_error=None,fill_value="extrapolate")

    newq = np.linspace(np.log(q.min()),np.log(q.max()),1000)
    newy = gaussian_filter1d(np.log(z(np.exp(newq))),200,mode = 'nearest')
    # newy = np.log(z(np.exp(newq)))
    newg = interp1d(newq,newy,bounds_error=None,fill_value="extrapolate")
    
    def h(x):
        return np.exp(newg(np.log(x)))
    return h

def LinearLinear_convexhull(equator):
    q = equator.q
    y = equator.values


    low = q < 0.019
    mid = np.logical_and(q>0.034,q<0.036)
    high = q> 0.05
    bool = np.logical_or(low,mid)
    bool = np.logical_or(bool,high)

    points = np.array([q[bool],y[bool]]).transpose()
    hull = ConvexHull(points)
    hullpoints = np.array([[points[vertex, 0], points[vertex, 1]] for vertex in hull.vertices ])
    h = interp1d(hullpoints[:,0],hullpoints[:,1],bounds_error=False)

    return h
