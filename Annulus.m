function X = Annulus( n, p )
% Uniformly sample n points from the Annulus.  Set in ambient space R^p.
%   INPUT
%       n  : Number of points.
%       p  : Dimension of ambient Euclidean space (>= 2).
%   OUTPUT
%       X  : Data matrix (n x p).
% Written by John Malik on 2018.13.10, john.malik@duke.edu

switch nargin
    case 1
        p = 2;
    case 0
        error('Select a number of points to sample.')
end

rho = nan(n, 1);
i = 1;
inner = sqrt(8) / 3;
while i <= n
    xvec = sqrt(rand(1));
    if xvec < 1 / 3 || xvec >= inner
        rho(i) = xvec;
        i = i + 1;
    else
        continue
    end
end
r = sort(rho);
t = 2 * pi * rand(n, 1);
x = r .* cos(t);
y = r .* sin(t);
X = [x, y];

if p > 2
    X = X * transpose(orth(randn(p, 2)));
end


end

