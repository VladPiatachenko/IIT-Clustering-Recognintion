% Завантаження даних
data = load('data24.txt');  % data має розмірність 200x2

% Побудова точкового графіка вхідних даних
figure;
scatter(data(:,1), data(:,2), 'filled');
title('Вхідні дані (точковий графік)');
xlabel('X координата');
ylabel('Y координата');
grid on;

% Параметри карти
OutputSizeX = 10; % Розмір по X
OutputSizeY = 10; % Розмір по Y
learningRate = 0.1; % Початковий коефіцієнт навчання
radius = max(OutputSizeX, OutputSizeY) / 2; % Початковий радіус околиці
numEpochs = 100; % Кількість епох навчання

% Ініціалізація ваг для кожного нейрона в карті випадковими значеннями в межах [-0.1, 0.1]
weights = -0.1 + (0.2) * rand(OutputSizeX, OutputSizeY, 2);

for epoch = 1:numEpochs
    for i = 1:size(data, 1)
        % Вибір поточного вхідного вектора
        inputVector = data(i, :);
        
        % Розрахунок відстаней між вхідним вектором і кожним нейроном карти
        distances = sqrt(sum((reshape(weights, [], 2) - repmat(inputVector, OutputSizeX * OutputSizeY, 1)).^2, 2));
        distances = reshape(distances, OutputSizeX, OutputSizeY);

        % Знаходження "виграшного" нейрона (нейрон з найменшою відстанню)
        [minval, winningIdx] = min(distances(:));
        [winX, winY] = ind2sub([OutputSizeX, OutputSizeY], winningIdx);

        % Зменшення радіуса околиці і швидкості навчання
        currRadius = radius * exp(-epoch / numEpochs);
        currLearningRate = learningRate * exp(-epoch / numEpochs);

        % Коригування ваг у межах околиці виграшного нейрона
        for x = 1:OutputSizeX
            for y = 1:OutputSizeY
                % Відстань від поточного нейрона до виграшного
                distToWinner = sqrt((x - winX)^2 + (y - winY)^2);
                
                % Якщо нейрон у межах околиці, скоригувати його ваги
                if distToWinner <= currRadius
                    influence = exp(-distToWinner^2 / (2 * currRadius^2));
                    % Виправлене додавання ваг з однаковими розмірностями
                    weights(x, y, :) = reshape(weights(x, y, :), 1, []) + currLearningRate * influence * (inputVector - reshape(weights(x, y, :), 1, []));
                end
            end
        end
    end
end

% Побудова карти ваг після навчання
figure;
scatter(data(:,1), data(:,2), 'filled'); % Вхідні дані
hold on;

% Візуалізація ваг кожного нейрона як точки на графіку
for x = 1:OutputSizeX
    for y = 1:OutputSizeY
        plot(weights(x, y, 1), weights(x, y, 2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
end

title('Самоорганізована карта Кохонена');
xlabel('X координата');
ylabel('Y координата');
grid on;
