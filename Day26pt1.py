## Learning CNN - Convolutional Neural Network ##

# Pixel -> each pixel is a number. For e.g -> Black = 0, white=255, everything in between is grey.
# Kernel -> is a small filter. a small grid of 3x3 values. Slides across the image will doing math.
# Convolution -> It slides across the kernel. At each position multiplies the kernel no. w/ the image numbers underneath and then adds all together
# to get one output no. This decreases the image quality from 9 pixels to 1 pixel. 9:1 ratio for decrease in pixel quality.

# Reason why we use Convolution is that different kernels detect different features. Some detect edges, some detect curves, some detect corners.

# Downsampling -> After convolution the output image is smaller. 9 pixels go in & 1 number comes out per position. The information is summarized not lost.
# Pooling -> It reduces the image size even further after convolution. Keeps only the most imp features. Makes the network faster and prevents overfitting.

## Summary ##

# CNN slides kernels across the image. Detects small patterns e.g edges and curves. Combines them into bigger patterns until it recognizes the whole image.
