
def high_pass_gaussian(size, sigma):
    P, Q = size
    x = np.arange(-Q//2, Q//2)
    y = np.arange(-P//2, P//2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    hpf = 1 - np.exp(-(D**2) / (2 * sigma**2))
    return hpf

sigma = 20
hpf = high_pass_gaussian(image.shape, sigma)

# Apply the filter to the shifted FFT of the image
filtered_fft = fft_image_shifted * hpf

# Inverse FFT to get the spatial domain image after filtering
filtered_image = np.abs(fftpack.ifft2(fftpack.ifftshift(filtered_fft)))

# Perform high frequency emphasis with a = 0.6 and b = 2
a = 0.6
b = 2
emphasized_image = a * image + b * filtered_image

# Display the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# High-pass Gaussian filter
axs[0].imshow(hpf, cmap='gray')
axs[0].set_title('High-pass Gaussian Filter')

# After filtering in frequency domain
axs[1].imshow(np.log(np.abs(filtered_fft) + 1), cmap='gray')
axs[1].set_title('After Applying High-pass Filter')

# High frequency emphasis result
axs[2].imshow(emphasized_image, cmap='gray')
axs[2].set_title('High Frequency Emphasis Result')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()