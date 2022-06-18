

from tatouage import tatouage
watermarking = tatouage()
watermarking.tato(img="lena.jpg", path_save=None)
watermarking.extracted(image_path="test_lena.jpg",extracted_watermark_path = None)