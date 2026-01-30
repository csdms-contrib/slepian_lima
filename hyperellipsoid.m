function [e,V]=hyperellipsoid(m,C,d,p);
% E=HYPERELLIPSOID(M,C,D,P);
%
% Calculate an (error) ellipsoid for any dimension d with scale parameter p
% (confidence level) given a central point m (mean or truth) and covariance
% matrix C 
%
% INPUTS
%
% m     center of the ellipsoid; may be the mean or true value expected from a
%       point cloud of estimates
% C     covariance matrix
% d     dimension; if not specified, default is to find the size of C
% p     the confidence level of the ellipsoid; with d, defines the scale 
%       parameter; defaults to 95%
%
% OUTPUTS
%
% e     a cell array for plotting
% V     eigenvector
%
% EXAMPLES
%
% e=hyperellipsoid(mean(thhats),cov(thhats),size(thhats,1),0.95);
%
% % Check the algebra with some fake values
% d=4;m=ones(d,1);
% C=triu(randn(d),1);C=C+eye(d)+C';
% e=hyperellipsoid(m,C,d,[]);
% clf;surf(e(:,:,1),e(:,:,2),e(:,:,3));
%
% SEE ALSO
%
% this is an extension to higher dimension of the problem described by this post
% https://raw.org/math/how-to-plot-a-covariance-error-ellipse/
%
% Last modified by olwalbert-at-princeton.edu, 06/13/2025

defval('d',size(C,1))
defval('p',0.95)

% Ellipses parameterized in radius by the standard deviation are sums of squares
% of Gaussians, and are thus distributed as chi-squared with degrees of freedom
% equivalent to the dimension

% We must determine the scale (rhs of ellipse definition) for the confidence
% given by p and degrees of freedom given by d
% if d==2; s=-d*log(1-p); end
s=chi2inv(p,d);

% For non-zero covariance, the ellipsoid is not aligned along the axes. We
% determine the alignment and spread by the scaled covariance matrix's 
% eigensystem
[V,Lam]=eig(C*s);

% Parameterize a d-1-sphere 
% If we were after a single parametric curve, the angular coordinates are just
t(1:d-2)=deal({linspace(0,pi)});
t(d-1)={linspace(0,2*pi)};
% Otherwise, we need to create a mesh for the angular coordinates
[T{1:d-1}]=ndgrid(t{:});

%r=repmat({ones(1,100)},1,d);
r=repmat({ones(100,100)},1,d);
for ind=1:d
  for jnd=1:ind-1
    %r{ind}=r{ind}.*sin(t{jnd});
    r{ind}=r{ind}.*sin(T{jnd});
  end
  if ind<d
    %r{ind}=r{ind}.*cos(t{ind});
    r{ind}=r{ind}.*cos(T{ind});
  end
end
% if d==2, diferm(r{1}-cos(t{1})); diferm(r{2}-sin(t{1}))
% if d==3, diferm(r{1}-cos(t{1})); diferm(r{2}-sin(t{1}).*cos(t{2}));
%          diferm(r{3}-sin(t{1}).*sin(t{2}));
% if d==4, diferm(r{1}-cos(t{1})); diferm(r{2}-sin(t{1}).*cos(t{2}));
%          diferm(r{3}-sin(t{1}).*sin(t{2}).*cos(t{3}));
%          diferm(r{4}-sin(t{1}).*sin(t{2}).*sin(t{3}));

% Scale and offset the origin-centered sphere to form the (hyper)ellipsoid
% Be careful of variable class and shape on this step.
%e=V*sqrt(Lam)*cat(1,r{:})+m;
e0=permute(tensorprod(V*sqrt(Lam),cat(3,r{:}),2,3),[2 3 1]);
e=e0+reshape(m,[repelem(1,d-1),d]);
