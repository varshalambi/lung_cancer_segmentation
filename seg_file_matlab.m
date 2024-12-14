lungImage = imread('/Users/varsha/Downloads/bs_project/data/lcd/data/bc/Bengin case (89).jpg');


V = im2single(lungImage);
%volumeViewer(V)


XY = V(:,:,3);
XZ = squeeze(V(256,:,:));

figure
imshow(XY,[],"Border","tight");
imshow(XZ,[],"Border","tight");


imageSegmenter(XY)


