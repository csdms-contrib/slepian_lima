function [ar,kappa]=anisotrosl(xs,params,a,mthd)
% [AR,kappa]=ANISOTROSL(XS,A,MTHD)
%
% Provides the lag handle for the clock-wise rotation by phi from positive y
% (North) that params.dydx(1) is aligned with to generate an anisotropic random
% field. We consider two methods of calculation, the first through matrix 
% multiplication of a rotation matrix with the original lags, and the second by 
% employing the dimensionless ellipsoidal norm described by Goff & Jordan
% (1988).
% OR method 3 transforms the two-dimensional wavenumbers following equation 22 of the
% proposal document.
%
% INPUTS:
%
% xs        The distance grid in space
% params    The grid parameters
% a         Anisotropy parameters
%           phi    clock-wise angle from North; 0 heading taken from the (m,1) to 
%                  the (1,1) element
%           alpha  stretch factor applied to the columns
% mthd      Calculation method
%           1      spatial rotation transformation
%           2      ellipsoidal norm following Goff & Jordan (1988)
%           3      deformed square wavenumber with stretch and rotation factors
%
% OUTPUTS:
%
% ar         Lags to provide to MATERNOSY whose handle is given to SGP within 
%            SIMULOSL to simulate an anistropic field
% kappa      The corresponding wavenumber grid
%
% SEE ALSO:
%
% SGP, SIMULOSL, MATERNOSY
%
% EXAMPLES:
%
% th0=[1 0.5 2];
% params=[];params.NyNx=[32 32];params.dydx=[4 1];params.blurs=Inf;
% phis=0:pi/6:2*pi;
% phi=phis(randi(size(phis,2)));
% a=[phi alpha];
% Cmn=@(h) maternosy(sqrt([h(:,1)*params.dydx(1)].^2+[h(:,2)*params.dydx(2)].^2,th0);
% Hx=sgpda(params,Cmn,2,a);
% clf;imagesc(Hx);title(sprintf('dydx=[%i %i]; phi=%0.2f',params.dydx,phi));
%
% Last modified by olwalbert-at-princeton.edu, 04/08/2025

defval('mthd',1)
defval('a',[0 1])
phi=a(1);
alpha=a(2);
% Notes: 
% Stein '99, p. 17: "We say (a random field) Z exhibits a geometric anisotropy 
% if there exists an invertible matrix V such that Z(Vx) is isotropic (Journel 
% and Huijbregts 1978, p. 177)" -> V is a linear transformation that takes an 
% ellipse to a circle in 2-D
switch mthd
  case 1
    R=[cos(phi) -sin(phi);...
       sin(phi)  cos(phi)];
    ar=R*xs';
    % ar=R*(xs*R)';
    % ar=((R*xs')'*R)';
    ar=ar'.*[1 alpha];
  case 2
    kappa=sqrt((k/ks).^2.*cos(phi-zs).^2+(k/kn).^2.*sin(phi-zs).^2);
  case 3
    params.blurs=-1;
    [~,~,~,kx,ky]=knums(params);
    ks{1}=kx-min(kx);
    ks{2}=ky-min(ky);
    % meshgrid
    d=2;
    [KS{1:d}]=ndgrid(ks{:});
    KS=reshape(cat(d+1,KS{:}),[],d);
    % Equation 22 of the NSF-DMS proposal
    R=[alpha.^(-1).*cos(phi).^2+alpha.*sin(phi).^2  (-alpha.^(-1)+alpha).*cos(phi).*sin(phi);
       (-alpha.^(-1)+alpha).*cos(phi).*sin(phi)     alpha.*cos(phi).^2+alpha.^(-1).*sin(phi).^2];
    kappa=(R*KS.').';
    XS(:,1)=params.NyNx(2)*params.dydx(2)/(2*pi)*kappa(:,1);
    XS(:,2)=params.NyNx(1)*params.dydx(1)/(2*pi)*kappa(:,2);
    % there is a small round-off issue
    XS=round(XS,6);
    % if R==I, this should pass
    % difer(xs-XS)
    ar=XS;
end
% figure()
% scatter(h(:,1),h(:,2))
