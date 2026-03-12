function generatedataset(numimgs)
path='J:/OneDrive/OneDrive - Univerza v Ljubljani/002ExperimentalData/underwater/curated/';
backgrounds = 'Background/';
objects ='Generated/BlueTang/fish%d';
masks = 'Generated/BlueTang/maskfish%d';
objectsidxs = 1:10;
dataset = 'traindataset/';
visual = 'traindatasetvisual/';

% Load reference color (best neutral colored background)
ref = imread([path,'referencecolor.jpg']);


% Background
list = dir([path,backgrounds,'*.jpg']);

for k=1:numimgs

    % Random choices for fish and background
    %---------------------------------------
    % Object
    object = objectsidxs(randi(length(objectsidxs)));
    objectfile = sprintf([path,objects,'.jpg'],object);
    maskfile = sprintf([path,masks,'.jpg'],object);    
    background = list(2+randi(length(list)-2)).name;
    % Load data
    bgr = imread([path,backgrounds,background]);
    obj = imread (objectfile);
    mask = imread (maskfile);
    bgr = imresize(bgr,[1080,1920],'bilinear');
    %refHD = imresize(ref,[1080,1920],'bilinear');
    illum = load([path,backgrounds,background,'.mat']);
    params = illum.params;
    [image, imagemarked, yoloBoundingBox] = processImage(bgr, obj, mask,  params);
    
    % get timestamp
    currentTime = datetime('now', 'TimeZone', 'UTC');
    epoch = datetime(1970, 1, 1, 'TimeZone', 'UTC');
    utcTimestamp = int64(1000*seconds(currentTime - epoch));
    

    % Create names
    [filepath,name,ext] = fileparts(background);    
    imagefilename = sprintf ('%s%s%s-fish%02d-%d.jpg',path,dataset,name,object,utcTimestamp);
    anotfilename = sprintf ('%s%s%s-fish%02d-%d.txt',path,dataset,name,object,utcTimestamp);
    visualfilename = sprintf ('%s%s%s-fish%02d-%d.jpg',path,visual,name,object,utcTimestamp);

    % save files
    imwrite (image,imagefilename,'Quality',90);
    imwrite (imagemarked,visualfilename,'Quality',90);
    fileId = fopen(anotfilename, 'w'); % Open a file for writing
    fprintf(fileId, '0 %f %f %f %f\n', yoloBoundingBox); % Write the class index and bounding box
    fclose(fileId); % Close the file
    fprintf(1,"Wrote image %d, name %s\n",k,imagefilename)

end

end


function [final, finalmarked, yoloBoundingBox] = processImage(bgr, obj, mask,  params)

% Fish image
fish = obj;

% Resize mask to match fish
s = size(fish);
s = s(1:2);
mask = rgb2gray(mask);
% Remove any compression artifacts
mask = double(mask>128);
mask = imresize (mask, s, 'nearest');

% dilate the mask so no black background comes through
se = strel('disk', 17);
mask = mask > 0.5;
mask = imdilate(mask, se);
mask = double (mask);


% Resize fish from 0.01*512 to 0.3*512
fishsize = randi([round(0.2*512),round(0.75*512)]);
fish = imresize(fish, [fishsize, fishsize],'bilinear');
mask = imresize(mask, [fishsize, fishsize],'nearest');

% Apply the color adjustment 50% probability
if randi(2)==2
    fish = imadjust(fish, ...
        [params(1), params(2), params(3); params(4), params(5), params(6)], ...
        [params(7), params(8), params(9); params(10), params(11), params(12)]);
end

% Rotate the fish randomly (between -45 and 45 degrees)
rotationAngle = randi([-45, 45]);
fish= imrotate(fish, rotationAngle, 'bilinear', 'crop');

% For mask we invert it before and after, so no need for filling
% corners!
mask = 1- mask;
mask = imrotate(mask, rotationAngle, 'nearest', 'crop');
mask = 1- mask;



% mirror the fish randomly along horizontal axis (12.5% probability)
if randi(8)==8
    fish = flipdim (fish,1);
    mask = flipdim (mask,1);
end

% mirror the fish randomly along vertical axis (50% probability)
if randi(2)==2
    fish = flipdim (fish,2);
    mask = flipdim (mask,2);
end

% squeeze randomly in each direction, probability of squeeze is 50%
choice = randi(4);
if choice==3
    s = size (fish);
    s = s(1:2);
    s(1)=s(1)*(0.2+(rand*0.8));
    fish = imresize (fish,s,'bilinear');
    mask = imresize (mask,s,'nearest');
end
if choice==4
    s = size (fish);
    s(2)=s(2)*(0.2+(rand*0.8));
    s = s(1:2);
    fish = imresize (fish,s,'bilinear');
    mask = imresize (mask,s,'nearest');
end

% Add gaussian noise with random amplitude 0-20, blur and again random
% amplitude 0-5
amp1 = 10+10*rand;
amp2 = 2+3*rand;
fish = uint8(double(fish) + amp1*randn(size(fish),'double'));
fish = imgaussfilt(fish, 1.0,'Padding', 1);
fish = uint8(double(fish) + amp2*randn(size(fish),'double'));

% Get background region
background = bgr;
% Randomly choose 512x512 to 1024x1024 region to sample as background
wh = randi ([512,1024]);

% Randomly choose the position of background sampling region
[h_b, w_b, ~] = size(background);
x = randi([1, w_b-wh]);
y = randi([1, h_b-wh]);

% Sample/crop the background
background = background (y:y+wh-1, x:x+wh-1,:);

% Resize to 512x512
background = imresize (background, [512,512],'bilinear');

% Randomly choose the position of the fish within the background
[h_b, w_b, ~] = size(background);
w = size(fish, 2);
h = size(fish, 1);
x = randi([1, w_b-w]);
y = randi([1, h_b-h]);

% Extract the background the fize of fish
fishBackground = background((y + 1):(y + h), (x + 1):(x + w), 1:3);

% Measure the position of the bounding box from mask!
maskbw = mask < 0.5;
maskstats= regionprops(maskbw, 'BoundingBox');
largestArea = 0;
largestBoundingBox = [];
for k = 1:length(maskstats)
    bb = maskstats(k).BoundingBox; % [x y width height]
    area = bb(3) * bb(4); % Calculate area (width * height)
    if area > largestArea
        largestArea = area;
        largestBoundingBox = bb;
    end
end


bb = [largestBoundingBox(1)+x, largestBoundingBox(2)+y, largestBoundingBox(3:4)];


% blur the edges of the mask, 5x5 (sigma = 1) filter.
mask = imgaussfilt(mask, 2.0,'Padding', 1);

% Combine images
fishCombined = combineImages(fish, fishBackground, mask);
imshow (fishCombined);

final = background;
final((y + 1):(y + h), (x + 1):(x + w), 1:3) = fishCombined;

% Draw the bounding box
finalmarked = insertShape(final, 'rectangle', bb, 'ShapeColor', 'yellow', 'LineWidth', 3);

% Transform bounding box to yolo format
% Assuming 'bb' contains bounding box [x y width height]

imgWidth = size(final, 2);
imgHeight = size(final, 1);

% Convert to YOLO format [center_x center_y width height]
xCenter = (bb(1) + bb(3) / 2) / imgWidth;
yCenter = (bb(2) + bb(4) / 2) / imgHeight;
normWidth = bb(3) / imgWidth;
normHeight = bb(4) / imgHeight;

yoloBoundingBox = [xCenter, yCenter, normWidth, normHeight];


end

function combined = combineImages(fish, background, mask)
fishR = fish(:, :, 1);
fishG = fish(:, :, 2);
fishB = fish(:, :, 3);

backgroundR = background(:, :, 1);
backgroundG = background(:, :, 2);
backgroundB = background(:, :, 3);

combined(:, :, 1) = uint8(double(fishR) .* (1 - double(mask)) + double(backgroundR) .* double(mask));
combined(:, :, 2) = uint8(double(fishG) .* (1 - double(mask)) + double(backgroundG) .* double(mask));
combined(:, :, 3) = uint8(double(fishB) .* (1 - double(mask)) + double(backgroundB) .* double(mask));
end



