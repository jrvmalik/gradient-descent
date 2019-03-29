function [updates, state] = RMSprop(gradients, state)
%RMSPROP rmsprop optimization
%   Detailed explanation goes here

if nargin == 1
    state = struct;
end

if ~isfield(state, 'alpha')
    state.alpha = 1e-3;
end
if ~isfield(state, 'rho') 
    state.rho = 0.9;
end
if ~isfield(state, 'epsilon')
    state.epsilon = 1e-8;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'history')
    state.history = zeros(size(gradients));
end


state.history = state.rho * state.history + (1 - state.rho) * gradients.^2;
    
% update parameters
updates = gradients * state.alpha ./ sqrt(state.history + state.epsilon);

% update iteration number
state.iteration = state.iteration + 1;


end

