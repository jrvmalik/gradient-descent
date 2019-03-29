clear
close all

% Parameters
n = 2000;
p = 2;
X = Annulus(n, p)';
Y = sum(X.^2) < .5;
Y = [Y; ~Y];
classes = size(Y, 1);
nodes = 10;
c = 1e-1;
d = 1e-4;
epochs = 45;
batch = 20;

% Initialize Weights and Biases
W1 = c * randn(nodes, p);
B1 = d * randn(nodes, 1);
W2 = c * randn(nodes, nodes);
B2 = d * randn(nodes, 1);
W3 = c * randn(classes, nodes);
B3 = d * randn(classes, 1);

% ReLU Activation Function
g = @(z) max(z, 0);
dg = @(z) z > 0;

%% Train

% Initialize Optimizer
Op = struct;
Op.alpha = 0.001;
W1S = Op;
B1S = Op;
W2S = Op;
B2S = Op;
W3S = Op;
B3S = Op;

for ii = 1:epochs
    
    % Shuffle Data and Labels
    t = randperm(n);
    XX = X(:, t);
    YY = Y(:, t);
    
    for jj = 1:ceil(n / batch)
        
        % Forward Pass
        A0 = XX(:, (jj - 1) * batch + 1:min(jj * batch, n));
        Z1 = W1 * A0 + B1;
        A1 = g(Z1);
        Z2 = W2 * A1 + B2;
        A2 = g(Z2);
        Z3 = W3 * A2 + B3;
        A3 = g(Z3);
        
        % Back-Propogated Error
        D3 = (A3 - YY(:, (jj - 1) * batch + 1:min(jj * batch, n))) .* dg(Z3);
        D2 = (W3' * D3) .* dg(Z2);
        D1 = (W2' * D2) .* dg(Z1);
        
        % Gradients
        W1G = D1 * A0' / size(A0, 2);
        B1G = mean(D1, 2);
        W2G = D2 * A1' / size(A0, 2);
        B2G = mean(D2, 2);
        W3G = D3 * A2' / size(A0, 2);
        B3G = mean(D3, 2);
        
        % Gradient Descent Optimizer
        [W1U, W1S] = Adadelta(W1G, W1S);
        [B1U, B1S] = Adadelta(B1G, B1S);
        [W2U, W2S] = Adadelta(W2G, W2S);
        [B2U, B2S] = Adadelta(B2G, B2S);
        [W3U, W3S] = Adadelta(W3G, W3S);
        [B3U, B3S] = Adadelta(B3G, B3S);
        
        % Perform Updates
        W1 = W1 - W1U;
        B1 = B1 - B1U;
        W2 = W2 - W2U;
        B2 = B2 - B2U;
        W3 = W3 - W3U;
        B3 = B3 - B3U;
        
    end
    
    % Print Loss
    Z = g(W3 * g(W2 * g(W1 * X + B1) + B2) + B3);
    disp(['Loss: ' num2str(sum((Z(:) - Y(:)).^2))]);

    % Visualize as Function on Unit Square
    [x, y] = meshgrid(linspace(-1, 1, 200), linspace(-1, 1, 200));
    z = g(W3 * g(W2 * g(W1 * [x(:)'; y(:)'] + B1) + B2) + B3);
    mesh(x, y, reshape(z(1, :), [200, 200])); drawnow;
    
end

% Scatter Plot with Predictions
figure;
[~, id] = max(Z, [], 1);
scatter(X(1, :), X(2, :), 20, id, 'filled');

