import os,sys
import numpy as np
import pylab as plt
sys.path.append('..')
import FiberDiffraction

N = 6
centeri = 3.25
centerj = 3.75
xx,yy = np.meshgrid(np.arange(0,N)-centerj,np.arange(0,N+2)-centeri)

print(xx)

R = np.sqrt(xx**2 + yy**2)


R2 = FiberDiffraction.fiber_image.CenterByPadding(R,centeri,centerj,subpixel=False,interp='Default')
R3 = FiberDiffraction.fiber_image.CenterByPadding(R,centeri,centerj,subpixel=True,interp='Default')


fig,ax = plt.subplots(1,3)
ax[0].imshow(R)
ax[0].set_title('Before')
ax[1].imshow(R2)
ax[1].set_title('Centering without subpixel')
ax[2].imshow(R3)
ax[2].set_title('Centering with subpixel')
plt.show()