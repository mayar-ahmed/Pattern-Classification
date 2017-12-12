from skimage import feature, exposure
from matplotlib import pyplot
import numpy


def hog(images):
    data = []
    for image in images:
        (feat, hog_image) = feature.hog(image, orientations=9,
                                        pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2),
                                        block_norm='L2',
                                        visualise=True)
        '''fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=pyplot.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=pyplot.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        pyplot.show()'''

        data.append(feat)
        print(feat.shape)
    numpy.stack(data)
    return data
