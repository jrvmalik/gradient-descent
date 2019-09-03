clear
close all

% train a nonlinear RNN to add binary numbers
% http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
% error in line 6 of bptt: change z to u from 6 on

% Written by John Malik on 2019.9.3   john.malik@duke.edu

% length of binary sequence
len = 7;

% number of sequences to train on
n = 1e3;

% generate pairs of numbers and their sums
vv = randi([0 sum(pow2(0:len-2))], n, 2);
yy = sum(vv, 2);

% convert to binary and flip to read left to right
v = zeros(2, len, n);
for i = 1:2
    v(i, :, :) = fliplr(dec2bin(vv(:, i), len))' - 48;
end
y(1, :, :) = fliplr(dec2bin(yy, len))' - 48;

% dimensions
vdim = size(v, 1); % number of input variables
hdim = 10; % number of state variables (hidden nodes)
odim = size(y, 1); % number of output variables
mem = size(v, 2); % length of sequence (T)

% initialize weights
init_wts = @(row, col) (2 * rand(row, col) - 1) * sqrt(6 / (row + col));
Whv = init_wts(hdim, vdim);
Whh = init_wts(hdim, hdim);
Woh = init_wts(odim, hdim);

% initialize biases
bh = init_wts(hdim, 1);
bo = init_wts(odim, 1);

% initial state
h0 = init_wts(hdim, 1);

% initialize optimizer with learning rate
rho = 1e-2;
WhvS.alpha = rho;
WhhS.alpha = rho;
WohS.alpha = rho;
bhS.alpha = rho;
boS.alpha = rho;
h0S.alpha = rho;

% number of steps
steps = 2e3;

% batch size
batch = 20;

% flatten matrix helper
flt = @(z) reshape(z, size(z, 1), batch);

% store training loss
E = zeros(steps, 1);

% activation functions
e = @(z) tanh(z);
de = @(z) sech(z).^2;
g = @(z) 1 ./ (1 + exp(-z));
dg = @(z) exp(-z) ./ (1 + exp(-z)).^2;



% minibatch gradient descent optimization
for tt = 1:steps
    
    % pick a random batch
    j = randi(n, batch, 1);
    
    % calculate hidden states
    u = zeros(hdim, mem, batch);
    h = zeros(hdim, mem, batch);
    u(:, 1, :) = Whv * flt(v(:, 1, j)) + Whh * h0;
    h(:, 1, :) = e(u(:, 1, :));
    for i = 2:mem
        u(:, i, :) = Whv * flt(v(:, i, j)) + Whh * flt(h(:, i - 1, :));
        h(:, i, :) = e(u(:, i, :));
    end
    
    % calculate output
    o = zeros(odim, mem, batch);
    for jj = 1:batch
        o(:, :, jj) = Woh * h(:, :, jj);
    end
    z = g(o);
    
    % loss per input pair
    E(tt) = sum((z - y(:, :, j)).^2, 'all') / batch;
    
    % initialize gradients
    dWhv = zeros(size(Whv));
    dWhh = zeros(size(Whh));
    dWoh = zeros(size(Woh));
    dbh = zeros(size(bh));
    dbo = zeros(size(bo));
    dh = zeros(size(h));
    du = zeros(size(u));
    do = zeros(size(o));
    
    % BPTT
    for t = mem:-1:1
        
        do(:, t, :) = dg(o(:, t, :)) .* (2 * (z(:, t, :) - y(:, t, j)));
        dbo = dbo + mean(do(:, t, :), 3);
        dWoh = dWoh + flt(do(:, t, :)) * flt(h(:, t, :))';
        dh(:, t, :) = flt(dh(:, t, :)) + Woh' * flt(do(:, t, :));
        du(:, t, :) = de(u(:, t, :)) .* dh(:, t, :);
        dWhv = dWhv + flt(du(:, t, :)) * flt(v(:, t, j))';
        dbh = dbh + mean(du(:, t, :), 3);
        
        if t > 1
            dWhh = dWhh + flt(du(:, t, :)) * flt(h(:, t - 1, :))';
            dh(:, t - 1, :) = Whh' * flt(du(:, t, :));
        else
            dWhh = dWhh + flt(du(:, t, :)) * repmat(h0', [batch, 1]);
            dh0 = mean(Whh' * flt(du(:, t, :)), 2);
        end
        
    end
    
    % compute updates
    [WhvU, WhvS] = RAdam(dWhv, WhvS);
    [WhhU, WhhS] = RAdam(dWhh, WhhS);
    [WohU, WohS] = RAdam(dWoh, WohS);
    [bhU, bhS] = RAdam(dbh, bhS);
    [boU, boS] = RAdam(dbo, boS);
    [h0U, h0S] = RAdam(dh0, h0S);
    
    % apply updates
    Whv = Whv - WhvU;
    Whh = Whh - WhhU;
    Woh = Woh - WohU;
    bh = bh - bhU;
    bo = bo - boU;
    h0 = h0 - h0U;
    
end

%% test on new pairs of binary sequences

clear v y

% number of test examples
n = 1e3;

% generate pairs of numbers and their sums
vv = randi([0 sum(pow2(0:len-2))], n, 2);
yy = sum(vv, 2);

% convert to binary and flip to read left to right
v = zeros(2, len, n);
for i = 1:2
    v(i, :, :) = fliplr(dec2bin(vv(:, i), len))' - 48;
end
y(1, :, :) = fliplr(dec2bin(yy, len))' - 48;

% pass each point one at a time
batch = 1;

% flatten matrix helper
flt = @(z) reshape(z, size(z, 1), batch);

% prediction
yhat = zeros(size(y));
for j = 1:n
    
    % calculate hidden states
    u = zeros(hdim, mem, batch);
    h = zeros(hdim, mem, batch);
    u(:, 1, :) = Whv * flt(v(:, 1, j)) + Whh * h0;
    h(:, 1, :) = e(u(:, 1, :));
    for i = 2:mem
        u(:, i, :) = Whv * flt(v(:, i, j)) + Whh * flt(h(:, i - 1, :));
        h(:, i, :) = e(u(:, i, :));
    end
    
    % calculate output
    o = zeros(odim, mem, batch);
    for jj = 1:batch
        o(:, :, jj) = Woh * h(:, :, jj);
    end
    z = g(o);
    
    % create binary vector
    yhat(:, :, j) = round(z);
    
end

% accuracy
ac = mean(all(yhat == y, 2));

figure; plot(E); ylabel('Training Loss') 
xlabel('Training Step'); title(['Test Accuracy: ' num2str(100*ac)]);