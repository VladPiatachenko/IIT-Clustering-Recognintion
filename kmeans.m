% Завантажуємо дані
X = [8.1 8.4; 8.3 7.9; 7.7 7.8; 7.9 7.8; 7.4 8.6; 8.5 8.2; 6.8 7.1; 7.4 6.9;
     8.4 8.9; 7.6 8.8; 7.5 7.5; 7.8 6.8; 7.3 7.1; 8.7 8.0; 6.9 8.8; 9.0 8.0;
     6.0 8.5; 8.7 7.1; 7.1 7.7; 7.4 5.9];

% Визначаємо кількість кластерів
num_clusters = 3;

% Виконуємо кластеризацію методом k-means
[cluster_idx, cluster_centers] = kmeans(X, num_clusters);

% Виводимо кількість елементів у кожному кластері
num_elements_in_cluster = histc(cluster_idx, 1:num_clusters);
disp('Кількість елементів у кожному кластері:');
disp(num_elements_in_cluster);

% Візуалізація кластерів
figure;
hold on;

% Визначаємо кольори для кожного кластеру
colors = lines(num_clusters);

for i = 1:num_clusters
    scatter(X(cluster_idx == i, 1), X(cluster_idx == i, 2), 100, colors(i,:), 'filled');
end

title('Кластеризація методом k-means');
xlabel('x1');
ylabel('x2');
legend(arrayfun(@(x) sprintf('Кластер %d', x), 1:num_clusters, 'UniformOutput', false));

hold off;

% Виводимо центри кластерів
disp('Центри кластерів:');
disp(cluster_centers);
