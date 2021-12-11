import numpy as np
import matplotlib.pyplot as plt


def my_fft(data,freq_len=40,x_input=np.zeros(10),kk=0,min_f=0,max_f=np.pi/3,isnorm=1):
    second_diff_input=np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input)<1e-10 :
        datat=np.squeeze(data)
        datat_fft = np.fft.fft(datat)
        ind2=range(freq_len)
        fft_coe=datat_fft[ind2]
        if isnorm==1:
            return_fft=np.absolute(fft_coe)
        else:
            return_fft=fft_coe
    else:
        return_fft=get_ft_multi(x_input,data,kk=kk,freq_len=freq_len,min_f=min_f,max_f=max_f,isnorm=isnorm)
    return return_fft

#NU DFT
def get_ft_multi(x_input,data,kk=0,freq_len=100,min_f=0,max_f=np.pi/3,isnorm=1):
    # x_input: sample x dim; data: sample x y_dim; kk: x_dim x k_sample
    # data is the y_input
    n=x_input.shape[1]
    if np.max(abs(kk))==0:
        k = np.linspace(min_f,max_f,num=freq_len,endpoint=True)
        kk = np.matmul(np.ones([n,1]),np.reshape(k,[1,-1]));

    tmp=np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
    if isnorm == 1:
        return_fft=np.absolute(tmp)
    else:
        return_fft=tmp
    return np.squeeze(return_fft)

# high dimension NUDFT
def highD_NUDFT(x_input, y_input, p1, freq_len=100, min_f=0,max_f=0.5, isnorm=1):
    # x_input: n x dim, p1: 1 x dim, k: freq_len x 1, y_input: n x y_dim
    n = x_input.shape[0]
    dim = x_input.shape[1]
    k = np.linspace(1/np.pi, max_f, num=freq_len, endpoint=True)
    #kp1 = np.matmul(np.reshape(k, (freq_len, 1)), np.reshape(p1, (1, -1))) # freq_len x n
    xp1 = np.matmul(x_input, p1.reshape([dim, 1])) # n x 1
    # different from the 1D, kk is too large to allocate (n x freq_len x n)
    # (60000 x 100 x 60000) with float64 element is too large
    # we have to compute the frequency for each k
    kxp1 = xp1.repeat(freq_len, axis=1) * k # (n x freq_len)
    return_fft = np.matmul(np.transpose(y_input), np.exp(-1J * 2 * np.pi * kxp1)) # (y_dim x freq_len)

    if isnorm==1:
        return_fft=np.absolute(return_fft)

    return k, np.transpose(np.squeeze(return_fft))

def get_difference(hk, yk):
    return np.linalg.norm(hk - yk, axis=1) / np.linalg.norm(yk, axis=1)
    #return np.abs((hk - yk) / yk)


def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=1,isShowPic=0):
    if isax==1:
        pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
#        ax.set_xlabel('step',fontsize=18)
#        ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic:
        pltm.show()
    else:
        pltm.close()

