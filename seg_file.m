% Load the lung CT scan image
lungImage = imread('/Users/varsha/Downloads/bs_project/data/lcd/data/bc/Bengin case (89).jpg');
% Convert the image to grayscale if it's not already
if size(lungImage, 3) > 1
    lungImageGray = rgb2gray(lungImage);
else
    lungImageGray = lungImage;
end

% Preprocessing: Enhance contrast
lungImageEnhanced = adapthisteq(lungImageGray);

% Segmentation: Thresholding
threshold = graythresh(lungImageEnhanced);
lungMask = imbinarize(lungImageEnhanced, threshold);

% Post-processing: Remove small regions and fill holes
lungMask = bwareaopen(lungMask, 1000); % Remove small objects
lungMask = imfill(lungMask, 'holes'); % Fill holes

% Visualize the segmented lung regions
imshowpair(lungImage, lungMask, 'montage');
title('Original Image vs. Segmented Lung Regions');