% plot network_conv.mat

load '../results/network_conv.mat'

figure,
hold on;
plot(baseline{1}.*100, 'LineWidth', 2);
plot(data_94{1}.*100, 'LineWidth', 2);
plot(data_62{1}.*100, 'LineWidth', 2);
plot(data_62_16{1}.*100, 'LineWidth', 2);
hold off;
ylabel('Evaluation Accuracy %')
xlabel('Epoch')

legend('baseline', 'setup 1', 'setup 2', 'setup 3')
grid on;



