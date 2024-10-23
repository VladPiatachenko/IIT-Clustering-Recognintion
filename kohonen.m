% Load data
X = [8.1 8.4; 8.3 7.9; 7.7 7.8; 7.9 7.8; 7.4 8.6; 8.5 8.2; 6.8 7.1; 7.4 6.9;
     8.4 8.9; 7.6 8.8; 7.5 7.5; 7.8 6.8; 7.3 7.1; 8.7 8.0; 6.9 8.8; 9.0 8.0;
     6.0 8.5; 8.7 7.1; 7.1 7.7; 7.4 5.9];

% Network settings
num_neurons = 3; % Number of neurons (clusters)
input_dim = size(X, 2); % Input dimension
learning_rate = 0.1; % Learning rate
epochs = 300; % Number of epochs

% Initialize weights
weights = rand(num_neurons, input_dim); % Weights should match the dimension of X

% Normalize weights
for i = 1:size(weights, 1)
    weights(i, :) = weights(i, :) / norm(weights(i, :)); % Normalize each weight vector
end

% Train Kohonen network (Self-Organizing Map)
winner_idx_array = zeros(size(X, 1), 1); % Array to store the winning neuron index for each input

for epoch = 1:epochs
    for i = 1:size(X, 1)
        % Select current input vector
        x = X(i, :);
        
        % Calculate distances between input and all neurons (Euclidean distance)
        distances = sum((weights - repmat(x, size(weights, 1), 1)).^2, 2); % Use repmat for manual expansion
        [min_val, winner_idx] = min(distances); % Find the winning neuron
        
        % Store the index of the winning neuron
        winner_idx_array(i) = winner_idx;

        % Update weights of the winning neuron
        weights(winner_idx, :) = weights(winner_idx, :) + learning_rate * (x - weights(winner_idx, :));
        
        % Update weights of neighboring neurons
        for j = 1:num_neurons
            if j ~= winner_idx
                % You can adjust the neighborhood function here
                % For simplicity, we'll use a linear decay based on distance from the winner
                distance_from_winner = abs(j - winner_idx);
                neighborhood_factor = max(0, (1 - distance_from_winner / num_neurons)); % Example neighborhood function
                weights(j, :) = weights(j, :) + (learning_rate * neighborhood_factor) * (x - weights(j, :));
            end
        end
    end
    
    % Decrease learning rate
    learning_rate = learning_rate * 0.9; % Ensure it decays over epochs
end

% Clustering results
cluster_centers = zeros(num_neurons, input_dim); 
for j = 1:num_neurons
    cluster_points = X(winner_idx_array == j, :); 
    if ~isempty(cluster_points)
        cluster_centers(j, :) = mean(cluster_points, 1); 
    end
end


figure;
hold on;
colors = lines(num_neurons);

for j = 1:num_neurons
    cluster_points = X(winner_idx_array == j, :); 
    scatter(cluster_points(:, 1), cluster_points(:, 2), 100, colors(j, :), 'filled', 'DisplayName', sprintf('Cluster %d', j)); 
end


scatter(cluster_centers(:, 1), cluster_centers(:, 2), 200, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'Cluster Centers');

title('Kohonen Self-Organizing Map Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
grid on;
hold off;

% Display the number of elements in each cluster
for j = 1:num_neurons
    num_elements = sum(winner_idx_array == j);
    fprintf('Number of elements in Cluster %d: %d\n', j, num_elements);
end
