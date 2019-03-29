function [updates, state] = DBD(gradients, state)
%DBD Delta-bar-delta optimization
%   Detailed explanation goes here

if nargin == 1
    state = struct;
end

if ~isfield(state, 'alpha')
    state.alpha = 1e-2;
end
if ~isfield(state, 'momentum') 
    state.momentum = 0.5;
end
if ~isfield(state, 'mini')
    state.mini = 0.01;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'updates')
    state.updates = zeros(size(gradients));
end
if ~isfield(state, 'gains')
    state.gains = ones(size(gradients));
end
if ~isfield(state, 'kappa')
    state.kappa = 0.2;
end
if ~isfield(state, 'phi')
    state.phi = 0.8;
end

% delta bar delta
dbd = sign(gradients) == sign(state.updates);

% decrease gains when moving in the opposite direction
state.gains(dbd) = state.gains(dbd) * state.phi;

% increase gains when moving in the same direction
state.gains(~dbd) = state.gains(~dbd) + state.kappa;

% clip gains from below
state.gains = max(state.gains, state.mini);
    
% update parameters using momentum term
updates = state.alpha * (state.gains .* gradients) - state.momentum * state.updates;

% notation
state.updates = -updates;  

% update iteration number
state.iteration = state.iteration + 1;


end

