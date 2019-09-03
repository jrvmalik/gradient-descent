function [updates, state] = Vanilla(gradients, state)
%VANILLA The most basic gradient descent
%   Detailed explanation goes here

if nargin == 1
    state = struct;
end

if ~isfield(state, 'alpha')
    state.alpha = 1e-1;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end

% compute updates
updates = state.alpha * gradients;

% update iteration number
state.iteration = state.iteration + 1;


end

