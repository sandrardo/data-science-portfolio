path='J:/OneDrive/OneDrive - Univerza v Ljubljani/002ExperimentalData/underwater/curated/';
backgrounds = 'Background/';
objects ='Generated/BlueTang/fish%d';
masks = 'Generated/BlueTang/maskfish%d';
objectsidxs = 1:10;

% Load reference color (best neutral colored background)
ref = imread([path,'referencecolor.jpg']);
list = dir([path,backgrounds,'*.jpg']);

refHD = imresize(ref,[1080,NaN],'bilinear');
numBins = 8;

% Go through all the background images to calculate illuminant
parfor n=3:length(list)
    background = list(n).name;

    bgr = imread([path,backgrounds,background]);
    bgrHD = imresize(bgr,[1080,NaN],'bilinear');

    hist3dtarget = dohistogram(bgrHD,8);

    % Initial guess for the parameters
    initialParams = [0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7];

    % Define bounds for the parameters (all between 0 and 1)
    lb = zeros(1, 12); % Lower bounds
    ub = ones(1, 12);  % Upper bounds

    % Run optimization
    options = optimset('fminsearch');
    options.MaxFunEvals = 1000;
    options.MaxIter = 1000;
    n
    [optimizedParams, fval] = fminsearch(@(params) objectiveFunction(params, refHD, hist3dtarget, numBins), initialParams, options);    
    fval
    saveresults ([path,backgrounds,background,'.mat'], optimizedParams,fval);

end

function saveresults (filename, params, fval)
    save (filename,'params','fval');
end

function distance = objectiveFunction(params, img, hist3dtarget, numBins)
    
    % Handling limits
    if any(params < 0) || any(params > 1)
        distance = 1e10; % A large number
        return;
    end
    % low values have to be lower than high values
    lows = [params(1:3),params(7:9)];
    highs = [params(4:6),params(10:12)];
    diffs = highs-lows;
    if any(diffs < 0)
        distance = 1e10; % A large number
        return;
    end
    


    % Apply imadjust with the given parameters
    adjustedImg = imadjust(img, ...
        [params(1), params(2), params(3); params(4), params(5), params(6)], ...
        [params(7), params(8), params(9); params(10), params(11), params(12)]);
    %params
    % Calculate the 3D histogram of the adjusted image
    hist3Dopt = dohistogram (adjustedImg, numBins); % Use the function from earlier

    % Calculate the Bhattacharyya distance
    distance = bhattacharyyaDist(hist3dtarget, hist3Dopt);
end

function hist3D = dohistogram (img, numBins)
[rows, cols, ~] = size(img);
pixels = double(reshape(img, rows * cols, 3));
scaledPixels = floor(pixels * (numBins / 256)) + 1;

hist3D = zeros(numBins, numBins, numBins);
for i = 1:size(scaledPixels, 1)
    R = scaledPixels(i, 1);
    G = scaledPixels(i, 2);
    B = scaledPixels(i, 3);
    hist3D(R, G, B) = hist3D(R, G, B) + 1;
end
end

function distance = bhattacharyyaDist(hist1, hist2)
    % Ensure histograms are normalized
    hist1 = hist1 / sum(hist1(:));
    hist2 = hist2 / sum(hist2(:));
    
    % Compute Bhattacharyya coefficient
    BC = sum(sum(sum(sqrt(hist1 .* hist2))));
    
    % Compute Bhattacharyya distance
    distance = -log(BC);
end

function transformedParams = transformParams(rawParams)
    transformedParams = 1 ./ (1 + exp(-rawParams));
end
