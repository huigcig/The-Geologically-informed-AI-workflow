import sys

from java.awt import *
from java.io import *
from java.nio import *
from java.lang import *
from javax.swing import *

from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.util import *
from edu.mines.jtk.util.ArrayMath import *
from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.interp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.ogl.Gl import *
from edu.mines.jtk.sgl import *
from edu.mines.jtk.util import *
from edu.mines.jtk.util.ArrayMath import *

from strat_skeleton import *
_dataDir = "../../data/"

def main(args):
  for ni in range(1):
    if ni%100==0: print "synthetic --> ", ni
    goLinearity(ni)
#   for ni in range(1):
#     if ni%100==0: print "F3 --> ", ni
#     goLinearity_F3(ni)
#   for ni in range(1):
#     if ni%100==0: print "Poseidon --> ", ni
#     goLinearity_Poseidon(ni)
#   for ni in range(1):
#     if ni%100==0: print "Beagle --> ", ni
#     goLinearity_Beagle(ni)

def goLinearity(num):
  ffile = str(num)
  global s1,s2
  global n1,n2
  global seismicDir
  seismicDir = _dataDir+"synthetic/seis/"
  saveliDir = _dataDir+"synthetic/linearity/"
  saveslDir = _dataDir+"synthetic/slope/"
  savenoDir = _dataDir+"synthetic/normal/"
  savecigDir = _dataDir+"synthetic/cigfacies/"
  savecigenDir = _dataDir+"synthetic/cigfacies_enhance/"
  s1 = Sampling(300,1.0,0.0)
  s2 = Sampling(520,1.0,0.0)
  n1,n2 = s1.count,s2.count

  f = readImageL(ffile) # seismic image
  p2 = zerofloat(n1,n2)
  wp = zerofloat(n1,n2)
  u11 = zerofloat(n1,n2)
  u12 = zerofloat(n1,n2)
  sigma1,sigma2,pmax=8.0,2.0,2.0
  lsf = LocalSlopeFinder(sigma1,sigma2,pmax)
  lsf.findSlopes_normal(f,p2,wp,u11,u12) # estimate slopes(p2) and linearity(wp) and normal vector  
  u1 = zerofloat(n1,n2,2)
  u1[0] = u11
  u1[1] = u12
  writeImageL(saveliDir,ffile,wp) # linearity
  writeImageL(saveslDir,ffile,p2) # slope
  writeImageL(savenoDir,ffile,u1) # normal

  # localsmoothingfilter guided by structure (provided by xmwu and Hale)
  f = readImageL(ffile)
  lof = LocalOrientFilter(6,3)
  ets = lof.applyForTensors(f)
  ets.setEigenvalues(0.1,1)
  lsf = LocalSmoothingFilter()
  lsf.apply(ets,10,f,f)
  cigfacies = find_2D_peaks(f,wp) # compute the peak data (based on seis & linearity)
  writeImageL(savecigDir,ffile,cigfacies)
    
  # If you want to enhance the sampling rate in high slope angle, please use the cigfacies_enhance version, 
  # The Angle threshold can be manually setting in the function below
  cigfacies_enhance = find_2D_peaks_enhance(f,wp,p2) # compute the peak data (based on seis & linearity & slope) 
  writeImageL(savecigenDir,ffile,cigfacies_enhance)

def goLinearity_F3(num):
  ffile = str(num)
  global s1,s2
  global n1,n2
  global seismicDir
  seismicDir = _dataDir+"F3/seis/"
  saveliDir = _dataDir+"F3/linearity/"
  saveslDir = _dataDir+"F3/slope/"
  savenoDir = _dataDir+"F3/normal/"
  savecigDir = _dataDir+"F3/cigfacies/"
  savecigenDir = _dataDir+"F3/cigfacies_enhance/"
  s1 = Sampling(300,1.0,0.0)
  s2 = Sampling(476,1.0,0.0)
  n1,n2 = s1.count,s2.count
  f = readImageL(ffile)
  p2 = zerofloat(n1,n2)
  wp = zerofloat(n1,n2)
  u11 = zerofloat(n1,n2)
  u12 = zerofloat(n1,n2)
  sigma1,sigma2,pmax=8.0,2.0,2.0
  lsf = LocalSlopeFinder(sigma1,sigma2,pmax)
#   lsf.findSlopes(f,p2,wp)
  lsf.findSlopes_normal(f,p2,wp,u11,u12)  
  u1 = zerofloat(n1,n2,2)
  u1[0] = u11
  u1[1] = u12
  writeImageL(saveliDir,ffile,wp)
  writeImageL(saveslDir,ffile,p2)
  writeImageL(savenoDir,ffile,u1)
  f = readImageL(ffile)
  lof = LocalOrientFilter(6,3)
  ets = lof.applyForTensors(f)
  ets.setEigenvalues(0.1,1)
  lsf = LocalSmoothingFilter()
  lsf.apply(ets,10,f,f)
  cigfacies = find_2D_peaks(f,wp)
  writeImageL(savecigDir,ffile,cigfacies)
  cigfacies_enhance = find_2D_peaks_enhance(f,wp,p2)
  writeImageL(savecigenDir,ffile,cigfacies_enhance)


def goLinearity_Poseidon(num):
  ffile = str(num)
  global s1,s2
  global n1,n2
  global seismicDir
  seismicDir = _dataDir+"Poseidon/seis/"
  saveliDir = _dataDir+"Poseidon/linearity/"
  saveslDir = _dataDir+"Poseidon/slope/"
  savenoDir = _dataDir+"Poseidon/normal/"
  savecigDir = _dataDir+"Poseidon/cigfacies/"
  savecigenDir = _dataDir+"Poseidon/cigfacies_enhance/"
  s1 = Sampling(300,1.0,0.0)
  s2 = Sampling(568,1.0,0.0)
  n1,n2 = s1.count,s2.count
  f = readImageL(ffile)
  p2 = zerofloat(n1,n2)
  wp = zerofloat(n1,n2)
  u11 = zerofloat(n1,n2)
  u12 = zerofloat(n1,n2)
  sigma1,sigma2,pmax=8.0,2.0,2.0
  lsf = LocalSlopeFinder(sigma1,sigma2,pmax)
#   lsf.findSlopes(f,p2,wp)
  lsf.findSlopes_normal(f,p2,wp,u11,u12)   
  u1 = zerofloat(n1,n2,2)
  u1[0] = u11
  u1[1] = u12
  writeImageL(saveliDir,ffile,wp)
  writeImageL(saveslDir,ffile,p2)
  writeImageL(savenoDir,ffile,u1)
  f = readImageL(ffile)
  lof = LocalOrientFilter(6,3)
  ets = lof.applyForTensors(f)
  ets.setEigenvalues(0.1,1)
  lsf = LocalSmoothingFilter()
  lsf.apply(ets,10,f,f)
  cigfacies = find_2D_peaks(f,wp)
  writeImageL(savecigDir,ffile,cigfacies)
  cigfacies_enhance = find_2D_peaks_enhance(f,wp,p2)
  writeImageL(savecigenDir,ffile,cigfacies_enhance)

def goLinearity_Beagle(num):
  ffile = str(num)
  global s1,s2
  global n1,n2
  global seismicDir
  seismicDir = _dataDir+"Beagle/seis/"
  saveliDir = _dataDir+"Beagle/linearity/"
  saveslDir = _dataDir+"Beagle/slope/"
  savenoDir = _dataDir+"Beagle/normal/"
  savecigDir = _dataDir+"Beagle/cigfacies/"
  savecigenDir = _dataDir+"Beagle/cigfacies_enhance/"
  s1 = Sampling(300,1.0,0.0)
  s2 = Sampling(520,1.0,0.0)
  n1,n2 = s1.count,s2.count
  f = readImageL(ffile)
  p2 = zerofloat(n1,n2)
  wp = zerofloat(n1,n2)
  u11 = zerofloat(n1,n2)
  u12 = zerofloat(n1,n2)
  sigma1,sigma2,pmax=8.0,2.0,2.0
  lsf = LocalSlopeFinder(sigma1,sigma2,pmax)
#   lsf.findSlopes(f,p2,wp)
  lsf.findSlopes_normal(f,p2,wp,u11,u12)     
  u1 = zerofloat(n1,n2,2)
  u1[0] = u11
  u1[1] = u12
  writeImageL(saveliDir,ffile,wp)
  writeImageL(saveslDir,ffile,p2)
  writeImageL(savenoDir,ffile,u1)
  f = readImageL(ffile)
  lof = LocalOrientFilter(6,3)
  ets = lof.applyForTensors(f)
  ets.setEigenvalues(0.1,1)
  lsf = LocalSmoothingFilter()
  lsf.apply(ets,10,f,f)

  cigfacies = find_2D_peaks(f,wp)
  writeImageL(savecigDir,ffile,cigfacies)
  cigfacies_enhance = find_2D_peaks_enhance(f,wp,p2)
  writeImageL(savecigenDir,ffile,cigfacies_enhance)


############################################################################################################
## functions
def find_2D_peaks(fx,wp,threshold=None):
  if threshold is None:
    threshold = -100
  peaks = zerofloat(n1,n2)
  for i in range(n2):
    for j in range(1,n1-1):
      wpi = wp[i][j]
      fxi = fx[i][j]
      fx1 = fx[i][j-1]
      fx2 = fx[i][j+1]
      if fxi>=threshold and wpi>=0.5:
        if fxi >= fx1 and fxi >= fx2:
          peaks[i][j]=1 # only one pixel
#           peaks[i][j-1]=1
#           peaks[i][j+1]=1
  return peaks

def find_2D_peaks_enhance(fx,wp,p2,threshold=None):
  if threshold is None:
    threshold = -100
  slope_value = 0.2 # 1-->45; 0.577 --> 30; 0.365-->20; 0.177-->10
  peaks = zerofloat(n1,n2)
  for i in range(n2):
    for j in range(1,n1-1):
      wpi = wp[i][j]
      fxi = fx[i][j]
      spi = p2[i][j]
      fx1 = fx[i][j-1]
      fx2 = fx[i][j+1]
      if fxi>=threshold and wpi>=0.5:
        if fxi >= fx1 and fxi >= fx2:
          peaks[i][j]=1
          if abs(spi)>=slope_value:
#               peaks[i][j-1]=1
              peaks[i][j+1]=1 # two pixels or more wide in T direction
  return peaks


def gain(x):
  g = mul(x,x) 
  ref = RecursiveExponentialFilter(20.0)
  ref.apply1(g,g)
  y = zerofloat(n1,n2)
  div(x,sqrt(g),y)
  return y
#############################################################################
# utilities

def readImage(name):
  fileName = seismicDir+name+".dat"
  n1,n2 = s1.count,s2.count
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImageL(name):
  fileName = seismicDir+name+".dat"
  n1,n2 = s1.count,s2.count
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def writeImage(name,image):
  fileName = seismicDir+name+".dat"
  aos = ArrayOutputStream(fileName)
  aos.writeFloats(image)
  aos.close()
  return image

def writeImageL(saveDir,name,image):
  fileName = saveDir+name+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main)) 
run(main)
