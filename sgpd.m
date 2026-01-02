function varargout=sgpd(params,Cnmp,d)
% [f1,f2,xs]=SGPD(params,Cnmp,d)
%
% Simulation of stationary Gaussian field over a d-dimensional grid using the
% circulant embedding method. References consulted include Dietrich & Newsam
% 1997 doi: 10.1137/S1064827592240555, Rezgi & Elden 2011 
% doi: 10.1016/j.laa.2010.03.032, and Kroese & Botev 2015 
% doi: 10.1007/978-3-319-10064-7_12.
%
% INPUT:
%
% params  parameters for the data grid, i.e. at least
%           params.dydx with the grid spacings
%           params.NyNx with the grid dimensions
%         We retain the naming convention even if d~=2, however the number of 
%         elements will be consistent with d (e.g., in 1-dimensions, 
%         params.NyNx=[NY] and params.dydx=[DY], and in 3-dimensions, 
%         params.NyNx=[NY NX NZ] and params.dydx=[DY DX DZ]).
% Cnmp    VECTORIZED scalar function handle to the covariance function
%         with one d-vector input such that cov(x1_t1,...,xd_td)=Cnmp(t1-...-td)
%         is the covariance function of a d-dimensional stationary Gaussian 
%         field.
% d       The dimension of the field.
% 
% OUTPUT:
%
% f1,f2   Two statistically independent fields over the d-dimensional grid.
% xs      Cell array of vector(s) {x1,...,xd}.
%
% SEE ALSO:
%
% SGP
%
% EXAMPLES:
%
% % For 1-dimension,
% d=1; th=[1 0.5 2];
% params=[];params.NyNx=[10];params.dydx=[1];
% Cnmp=@(h) maternosy([h(:,1)*params.dydx(1)],th);
% [f1,f2,xs]=sgpd(params,Cnmp,1);
% plot(xs{1},f1);hold on;plot(xs{1},f2);legend('f1','f2');
%
% % For 2-dimensions,
% d=2; params=[];params.NyNx=[10 10];params.dydx=[1 1];
% Cnmp=@(h) maternosy(sqrt([h(:,1)*params.dydx(1)].^2+...
%                          [h(:,2)*params.dydx(2)].^2),th);
% [f1,f2,xs]=sgpd(params,Cnmp,d);
% % Compare to sgp output:
% [f1_,f2_,x1_,x2_]=sgp(params,Cnmp);
% diferm(size(f1)-size(f1_))
% imagesc(xs{1},xs{2},f1)
% 
% % For 3-dimensions,
% d=3; params=[];params.NyNx=[30 30 30];params.dydx=[1 1 1];
% Cnmp=@(h) maternosy(sqrt([h(:,1)*params.dydx(1)].^2+...
%                          [h(:,2)*params.dydx(2)].^2+...
%                          [h(:,3)*params.dydx(3)].^2),th);
% [f1,f2,xs]=sgpd(params,Cnmp,d);
% % Plot a single slice
% imagesc(xs{1},xs{2},f1(:,:,1))
% imagesc(xs{1},xs{3},squeeze(f1(:,1,:)))
% imagesc(xs{2},xs{3},squeeze(f1(1,:,:)))
% % Display in 3-dimensions by providing indices of slices to plot
% yi=[floor(params.NyNx(1)/2) params.NyNx(1)]; 
% xi=[floor(params.NyNx(2)/2) params.NyNx(2)];
% zi=[1                       floor(params.NyNx(3)/2)];
% slice(f1,xi,yi,zi)
%
% % For d-dimensions (selecting d randomly from 1 to 5 for the sake of example,
% % the size of the mesh required for positive definite embedding will be a 
% % limiting factor)
% d=randi(5); if d>3; th=[1 0.5 1]; end
% params=[];params.NyNx=repelem(30,d);params.dydx=repelem(1,d);
% if d==1; Cnmp=@(h) maternosy([h(:,1)*params.dydx(1)],th); else;
% Cnmp=@(h) maternosy(vecnorm([h.*params.dydx]',2)',th)'; end;
% [f1,f2,xs]=sgpd(params,Cnmp,d);
% if d>3 % display the last 3 dimensions as a cube
%   yi=[floor(params.NyNx(d-3)/2) params.NyNx(d-3)]; 
%   xi=[floor(params.NyNx(d-2)/2) params.NyNx(d-2)];
%   zi=[1                         floor(params.NyNx(d-1)/2)];
%   seldim=[repelem({1},d-3) repmat({':'},1,3)];
%   slice(squeeze(f1(seldim{:})),xi,yi,zi)
% end
% 
% % Correlated n-d fields created from slices of n+1-dimensional fields
% n=2; d=n+1; 
% params=[];params.NyNx=repelem(30,d);params.dydx=repelem(1,d);
% Cnmp=@(h) maternosy(vecnorm([h.*params.dydx]',2)',th)';
% [f1,f2,xs]=sgpd(params,Cnmp,d); 
% % Independent 2-dimensional fields:
% corrcoef(f1(:,:,1),f2(:,:,1))
% % Correlated 2-dimensional fields:
% corrcoef(f1(:,:,1),f1(:,:,2))
%
% Last modified by olwalbert-at-princeton.edu, 03/31/2025

% Create vectors for the number of samples in each dimension, ts, and the
% length, xs, of each dimension by incorporating the step-size
ts=arrayfun(@colon,repelem(0,size(params.NyNx,2)),params.NyNx-1,...
            'Uniform',false);
for k=1:d
  xs{k}=ts{k}(1:end).*params.dydx(k);
end

% Assemble a rectangular, regularly meshed sample grid, TS, upon which we 
% calculate the covariance matrix from
[TS{1:d}]=ndgrid(ts{:});
TS=reshape(cat(d+1,TS{:}),[],d);

% Calculate the covariance matrix elements that will be used to form the first 
% d-dim block of the circulant matrix, s. Since we are in the stationary Matern
% case and depend on lag distances, r is symmetric and negating input arguments
% for Cnmp will be redundant. If we ever go beyond this case, see Dietrich+1997
% eq. 20 for 2 dimensions.
r=reshape(Cnmp(TS),[params.NyNx 1]);
% For the sake of memory, clear the meshgrid
clear TS

% Assemble the block
s=r;
for ind=1:d
  sz=size(s,ind);
  seldim1  =repmat({':'},1,ind-1);
  seldimend=repmat({':'},1,d-ind);
  s(seldim1{:},params.NyNx(ind)+1:2*params.NyNx(ind)-1,seldimend{:})=...
      s(seldim1{:},sz:-1:2,seldimend{:});
end
% For the sake of memory, clear the covariance calculation
clear r 

% Compute the eigenvalues
lam=real(fftn(s))/prod(size(s));
if abs(min(lam(lam(:)<0)))>1e-15
  % You will need a larger grid
  error('Could not find positive definite embedding!')
else
  lam(lam(:)<0)=0; 
  lam=sqrt(lam);
end
% For the sake of memory, clear the block
sz=size(s);
clear s

% Generate a field with covariance given by the block circulant matrix
A=lam.*complex(randn([sz 1]),randn([sz 1]));
% For the sake of memory, clear the eigenvalues
clear lam
F=fftn(A);

% Extract a subblock with the desired covariance for the desired grid-size
subinds=arrayfun(@colon,repelem(1,size(params.NyNx,2)),params.NyNx,...
                 'Uniform',false);
F=F(subinds{:});

% Two independent fields with desired covariance
f1=real(F); 
f2=imag(F);

% Optional output
varns={f1,f2,xs};
varargout=varns(1:nargout);

% TODO: sgpd_demo
% Form entire matrix, not just the first row, and check that it is diagonalized
% by FFT.
