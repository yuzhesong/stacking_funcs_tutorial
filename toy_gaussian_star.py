import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
def log_like(pix, ave):
    ##Likelihood value of the entire ROI when it's a flat field.
    ##pix is the array containing the simulated photon counts in all the pixels.
    ##ave is the flat field noise/background photon count
    sum=0
    l=len(pix)
    for i in range(l):
        sum=sum+pix[i]*np.log(ave)
    return sum-len(pix)*ave

def log_like_f(pix, ave, fave=0):
    ##Calculating the likelihood value of the ROI.
    ##pix is the array containing the simulated photon counts in all the pixels.
    ##ave is the flat field noise/background photon count
    ##fave is the photon count of the source (normalization of the gaussian)
    
    ####Count # of pixels in rings in Gaussian star, circular Gaussian source with 0.5deg FWHM
    nring=np.pi*(2*np.linspace(0,9,10)+1)+0.5
    nring=nring.astype(int)
    nring=np.append([0],nring)
    nring[10]=nring[10]+3
    iring=nring
    for i in range (2,11):
        iring[i]=iring[i]+iring[i-1]
    nring=np.pi*(2*np.linspace(0,9,10)+1)+0.5
    nring=nring.astype(int)
    nring=np.append([0],nring)
    nring[10]=nring[10]+3
    cring=2*np.exp(-np.linspace(0,9,10)**2*0.01*np.log(2.)/0.25)
    ####Calculate the likelihood of the entire ROI, pixel by pixel. 
    sum=0
    l=len(pix)
    pl=len(nring)
    nnn=np.amax(iring)
    for i in range(0,pl-1):
        for j in range(iring[i],iring[i+1]):
            sum=sum+pix[j]*np.log(ave+fave*cring[i])-(ave+fave*cring[i])
    for i in range(nnn,l):
        sum=sum+pix[i]*np.log(ave)-ave
    return sum

def get_ROI(p, ac, fave=0):
    ####Count # of pixels in rings in Gaussian star, circular Gaussian source with 0.5deg FWHM
    nring=np.pi*(2*np.linspace(0,9,10)+1)+0.5
    nring=nring.astype(int)
    nring=np.append([0],nring)
    nring[10]=nring[10]+3
    iring=nring
    for i in range (2,11):
        iring[i]=iring[i]+iring[i-1]
    nring=np.pi*(2*np.linspace(0,9,10)+1)+0.5
    nring=nring.astype(int)
    nring=np.append([0],nring)
    nring[10]=nring[10]+3
    cring=2*np.exp(-np.linspace(0,9,10)**2*0.01*np.log(2.)/0.25)
    
    counts=np.random.poisson(lam=ac, size=int(p))
    pl=len(nring)
    for i in range(0,pl-1):
        counts[iring[i]:iring[i+1]]=np.random.poisson(lam=ac+fave*cring[i],size=nring[i+1])

    return counts

if __name__ == "__main__":
    nor = 1000 #number of ROIs to look at.
    nop = 317*2 #number of pixels in ROI
    g = 0.03
    
    average_counts=19.447910

    f_a=2*average_counts*g/(g+2)
    average_counts=2*average_counts/(g+2)
    f2=average_counts/50.


    TS=np.array([])
    TS2=np.array([])
    p_counts=np.array([])

    ###Constructing all ROIs and calculate log(L), then the TS value.
    for i in range(0,nor):
        
        p_counts=get_ROI(nop, average_counts, f_a)
        pc2=get_ROI(nop, average_counts, f2)
        
        logL1 = log_like_f(p_counts,average_counts)
        logL2 = log_like_f(p_counts,average_counts, f_a)
        
        logL3 = log_like_f(pc2,average_counts)
        logL4 = log_like_f(pc2,average_counts, f2)
        
        TS = np.append(TS, 2*(logL4-logL3))
        TS2= np.append(TS2,2*(logL2-logL1))
        
        
    fff2=plt.figure()
    sub3=fff2.add_subplot(111)
    ts_min=int(np.amin([TS, TS2]))-1
    ts_max=int(np.amax([TS, TS2]))+1

    bins=np.linspace(0, ts_max, ts_max+1)
    TS[TS<0]=0
    TS2[TS2<0]=0
    (nn,nnbins,nnpatch)=sub3.hist( [TS,TS2], bins, histtype='step', color=['blue','green'], label=['noise model', 'flare model'])
    
    #Calculate the null distribution from chi2. 
    nnbins=np.append(nnbins, np.arange(nnbins[-1]+1,nnbins[-1]+4))
    chi2int=np.zeros(len(nnbins))
    for i in range(len(nnbins)-1):
        I=quad(stats.chi2.pdf,nnbins[i],nnbins[i+1],args=(2,0,1))
        chi2int[i]=len(TS)*I[0]/2
    
    sub3.plot(nnbins, chi2int*2,drawstyle='steps-post',label=r"${\chi}^2/2$, dof=2",c='orange')
    
    sub3.set_xlabel('TS')
    sub3.set_yscale('log')
    sub3.set_xlim(0,ts_max+1)
    sub3.set_ylim(0.9,len(TS))
    sub3.set_title('distribution of TS')
    plt.legend(loc='upper right')
    fff2.savefig('noise_dist_1000_new.eps')

    print(np.amin(TS), np.amax(TS))
    print(np.average(p_counts))