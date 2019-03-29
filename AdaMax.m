function [updates, state] = AdaMax(gradients, state)
%ADAMAX Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    state = struct;
end

if ~isfield(state, 'beta1')
    state.beta1 = 0.9;
end
if ~isfield(state, 'beta2') 
    state.beta2 = 0.999;
end
if ~isfield(state, 'epsilon') 
    state.epsilon = 1e-8;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'm')
    state.m = zeros(size(gradients));
end
if ~isfield(state, 'u')
    state.u = zeros(size(gradients));
end
if ~isfield(state, 'alpha')
    state.alpha = 1e-2;
end

% update biased first moment estimate
state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;
    
% update biased second raw moment estimate
state.u = max(state.beta2 * state.u, abs(gradients));
    
% compute bias-corrected first moment estimate
mhat = state.m / (1 - state.beta1^state.iteration);
    
% update parameters
updates = state.alpha * mhat ./ (state.u + state.epsilon);

% update iteration number
state.iteration = state.iteration + 1;


end

