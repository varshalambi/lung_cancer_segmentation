imageFolder = '/Users/varsha/Downloads/bs_project/data/lcd/data/nc/';  % Ensure correct path

% Correct variable name for filePattern and jpegFiles if you are using .png files
filePattern = fullfile(imageFolder, '*.jpg');  % Corrected to '*.png' based on your path
imageFiles = dir(filePattern);

for k = 1:length(imageFiles)
    baseFileName = imageFiles(k).name;
    fullFileName = fullfile(imageFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);

    % Read Image File
    Selected_Image = imread(fullFileName);
    if isempty(Selected_Image)
        fprintf('Failed to read %s\n', fullFileName);
        continue;  % Skip this iteration if image failed to load
    end

    % Create a new figure for each image to avoid subplot overwriting
    figure;

    % Original Image
    subplot(3,3,1);
    imshow(Selected_Image);
    title('Selected Image');

    % Noise Removal
    greyscale_Method = rgb2gray(Selected_Image);
    % Assuming customfilter is a valid function that you have defined elsewhere
    median_filtering_Image = customfilter(greyscale_Method);
    subplot(3,3,2);
    imshow(median_filtering_Image);
    title('Noise Removed');

    % Convert to Binary Image
    binary_picture = imbinarize(median_filtering_Image, 0.2);
    subplot(3,3,3);
    imshow(binary_picture);
    title('Binary Image');
    
    %Create morphological structuring element
    se1 = strel('disk', 2); %creates a flat, disk-shaped structure,
    postOpenImage_1 = imopen(binary_picture, se1);
    subplot(3, 3, 4); %divides figure into rectangular panes
    imshow(postOpenImage_1); %show image
    title('Opening Image'); %image name
    
    %Create morphological structuring element
    se2 = strel('disk', 8); %creates a flat, disk-shaped structure,
    postOpenImage_2 = imopen(binary_picture, se2);
    inverted = ones(size(binary_picture));
    
    %Creates Inverted Picture
    invertedImage_1 = inverted - postOpenImage_1;
    invertedImage_2 = inverted - postOpenImage_2;
    subplot(3,3,5); %divides figure into rectangular panes
    imshow(invertedImage_1); %show image
    title('Inverted Picture'); %image name
    
    %Specify initial contour
    mask = zeros(size(invertedImage_1)); 
    mask(50:end-50,50:end-50) = 1;
    
    %Segments image into foreground and background
    bw_1 = activecontour(invertedImage_1, mask, 800); %800 iterations 
    bw_2 = activecontour(invertedImage_2, mask, 400); %400 iterations 
    
    %Create Combination Pictures with Inverted image and Contour
    mix_Image_1 = invertedImage_1 + bw_1;
    filter_mix_Image_1 = medfilt2(mat2gray(mix_Image_1),[5 5]); %Filtering
    mix_Image_2 = invertedImage_2 + bw_2;
    filter_mix_Image_2 = medfilt2(mat2gray(mix_Image_2),[7 7]); %Filtering
    
    %Black White Combination to Create Final Images
    final_2 = im2bw(filter_mix_Image_2, 0.6);
    final_1 = im2bw(filter_mix_Image_1, 0.6);
    pre_final = final_1; %transfer
    subplot(3,3,6); %divides figure into rectangular panes
    BW5 = imfill(pre_final,'holes');
    imshow(BW5); %show image
    title('Segmented'); %image name
    
    %Dispaly Final Image
    subplot(3,3,7); %divides figure into rectangular panes
    imshow(final_1); %show image
    title('Final Picture'); %image name
    
    
    %Find Circles within Image using Polarity and Sensitivity
    %change this section of code to add a mobilenet model
    %or the other approach to this image density gradient
    
    warning('off', 'all')
    circle_image = final_1;
    [centers,radii] = imfindcircles(circle_image,[1 9],'ObjectPolarity','dark','Sensitivity',0.88);
    viscircles(centers,radii,'EdgeColor','g'); % Circles Display Green
    display(size(centers, 1), ' Numbers of Circles');
    subplot(3,3,8); %divides figure into rectangular panes
    imshow(circle_image); %show image
    title('Circle Found Picture'); %image name
    
    
    segment_pic = final_2 - final_1; % Creates Segmented Image
    %Creates before Colour Image 
    pre_colour_pic = im2bw(medfilt2(mat2gray(segment_pic),[3 3]), 0.6);
    subplot(3,3,9); %divides figure into rectangular panes
    imshow(pre_colour_pic); %show image
    title('Circles Segmented'); %image name
    
    %Circles Segmented Circles
    [B] = bwboundaries(pre_colour_pic,'holes');
    hold on
    for k  = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2)
    drawnow;

    end
end
