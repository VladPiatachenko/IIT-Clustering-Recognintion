% Load and preprocess training images
x0 = imread('0.bmp'); % Class 0
x4 = imread('4.bmp'); % Class 4
x8 = imread('8.bmp'); % Class 7

% Convert images to grayscale if needed and check dimensions
[y0, y4, y8] = deal(double(x0(:,:,1)), double(x4(:,:,1)), double(x8(:,:,1)));

% Check dimensions
dims = [size(y0, 1), size(y0, 2); 
        size(y4, 1), size(y4, 2); 
        size(y8, 1), size(y8, 2)];

if any(diff(dims, [], 1) ~= 0)
    error('All loaded characters must have the same dimensions.');
end
% Reshape each image to be a column vector
y0 = y0(:); % Flatten to a 120x1 vector
y4 = y4(:); % Flatten to a 120x1 vector
y8 = y8(:); % Flatten to a 120x1 vector

% Create the training matrix P (120 rows for each pixel, 3 columns for observations)
P = [y0, y4, y8]; % This creates a 120x3 matrix for training

% Create class vector C for the 3 training samples
C = [1; 2; 3];  % Each image corresponds to a different class

% Convert class vector to target matrix T for training
T = ind2vec(C');    % T will be a 3x3 matrix for the three classes

% Define the min-max range for the features (pixels)
t = [zeros(120, 1), ones(120, 1)]; % Min value 0, Max value 255 for each of the 120 features

% Create and train the LVQ network
numHiddenNeurons = 10; % Initialize with 10 hidden neurons
net = newlvq(t, numHiddenNeurons, [0.33 0.33 0.34], 0.01); % Initialize the network
net.trainParam.epochs = 500; % Set the number of training epochs

% Train the network
net = train(net, P, T); % Train the network
%%%
hiddenNeuronCounts = [5, 10, 15, 20];
accuracies = zeros(length(hiddenNeuronCounts), 1); % To store accuracies

for i = 1:length(hiddenNeuronCounts)
   net = newlvq(t, hiddenNeuronCounts(i), [0.33 0.33 0.34], 0.01);
    net.trainParam.epochs = 500;
    net = train(net, P, T);
    
    output = sim(net, P);
    [val, predictedClasses] = max(output); % Get predicted classes
    
    accuracies(i) = sum(predictedClasses' == C) / length(C);
end
fprintf('Hidden Neurons\tAccuracy\n');
for i = 1:length(hiddenNeuronCounts)
    fprintf('%d\t\t%.2f%%\n', hiddenNeuronCounts(i), accuracies(i) * 100);
end
%%%
% Recognize characters using the trained network
output = sim(net, P); % Test with training data
[val, predictedClasses] = max(output);

% Display the results
fprintf('Predicted classes: %d, %d, %d\n', predictedClasses);
%%%
% Load and preprocess the image to recognize
newImage = imread('un4.bmp'); % Load the test image
newImage = double(newImage(:,:,1)); % Convert to grayscale if necessary

% Ensure the new image has the correct size
if size(newImage, 1) ~= 12 || size(newImage, 2) ~= 10
    error('The input image must be a 12x10 matrix (120 pixels).');
end

% Flatten the image to a column vector
newImage = newImage(:); % Convert to 120x1 vector

% Create input matrix for recognition (1 sample, 120 features)
inputMatrix = newImage'; % Transpose to have 1 row (1 sample) and 120 columns

% Classify the new image using the trained network
output = sim(net, inputMatrix'); % Use sim to get network output

% Ensure the output has the expected dimensions
if size(output, 1) ~= 3
    error('Output size mismatch: expected a 3x1 vector for 3 classes.');
end

% Determine the predicted class
[val, predictedClass] = max(output); % Get the index of the max value (class label)

% Display the result
fprintf('The predicted class for the image un4.bmp is: %d\n', predictedClass);
