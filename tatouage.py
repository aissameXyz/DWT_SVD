#Author: aissameXyz
#data: 18/6/2022
import numpy as np
import cv2
import pywt

class Composants():
    Coeffs = []
    U = None
    S = None
    V = None
class tatouage():
    #attributes
    """
    :param tato_path:
    :param ratio:
    :param wavelet:
    :param level:
    """
    def __init__(self, tato_path="testwatermark.jpg", ratio=0.1, wavelet="haar", level=2):
        self.level = level
        self.wavelet = wavelet
        self.ratio = ratio
        self.shape_tato= cv2.imread(tato_path, 0).shape
        self.W_composants = Composants()
        self.img_composants = Composants()
        self.W_composants.Coeffs, self.W_composants.U, \
        self.W_composants.S, self.W_composants.V = self.calculer(tato_path)
    def calculer(self, img):
        
      
        #on calcul les coefficients el les composantes SVD.
        if isinstance(img, str):
            img = cv2.imread(img, 0)
            #image est numpy array
        Coeffs = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)
        self.shape_LL = Coeffs[0].shape
        U, S, V = np.linalg.svd(Coeffs[0])
        return Coeffs, U, S, V
    def diag(self, s):

        #To recover the singular values to be a matrix.
        #:param s: a 1D numpy array
        
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S
    def recover(self, name):
       
        #To recover the image from the svd components and DWT

        composants = eval("self.{}_composants".format(name))
        s = eval("self.S_{}".format(name))
        composants.Coeffs[0] = composants.U.dot(self.diag(s)).dot(composants.V)
        return pywt.waverec2(composants.Coeffs, wavelet=self.wavelet)


    def tato(self, img="lena.jpg", path_save=None):
        #This is the main function for image watermarking.
        #:param img: image path or numpy array of the image.
        
        if not path_save:
            path_save =  "watermarked_" + img
        self.path_save = path_save
        self.img_composants.Coeffs, self.img_composants.U, \
        self.img_composants.S, self.img_composants.V = self.calculer(img)
        self.embed()
        img_rec = self.recover("img")
        cv2.imwrite(path_save, img_rec)
    

    def extracted(self, image_path=None, ratio=None, extracted_watermark_path = None):
    
        #Extracted the watermark from the given image.

        if not extracted_watermark_path:
            extracted_watermark_path = "tatouage_extrait.jpg"
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path,0)
        img = cv2.resize(img, self.shape_tato)
        img_composants = Composants()
        img_composants.Coefficients, img_composants.U, img_composants.S, img_composants.V = self.calculer(img)
        ratio_ = self.ratio if not ratio else ratio
        self.S_W = (img_composants.S - self.img_composants.S) / ratio_
        watermark_extracted = self.recover("W")
        cv2.imwrite(extracted_watermark_path, watermark_extracted)

    def embed(self):
        self.S_img = self.img_composants.S + self.ratio * self.W_composants.S * \
                                             (self.img_composants.S.max() / self.W_composants.S.max())
                                        

if __name__ == '__main__':
    tatouage = tatouage(level=3)
    tatouage.tato()
    tatouage.extracted()




   

        



