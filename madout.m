function [B,C,kflags]=madout(A,dim,kappa)
% [B,C,KFLAGS]=MADOUT(A,DIM,KAPPA)
%
% Identifies and removes outliers using a robust method of calculating modified
% Z-scores from the median absolute deviation (MAD) of sample A along the
% columns. Outliers are identified as modified Z-scores that exceed KAPPA in 
% absolute value.
%
% INPUTS:
%
% A       The original sample array
% kappa   Scaling parameter for choosing the bounds about the modified Z-score;
%         the recommended and defaulted value is 3.5 (see references below).
%
% OUTPUTS:
%
% B       The new array with outliers removed
% C       An array of the removed outliers
% kflags  An array of logicals for row indices of array A; 1's are retained in B
%
% SEE ALSO:
%
% MAD, THRESHOLDS, TRIMMIT
% 
% REFERENCES:
%
% Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
% Handle Outliers", The ASQC Basic References in Quality Control: Statistical
% Techniques, Edward F. Mykytka, Ph.D., Editor. 
%
% https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
%
% EXAMPLES:
%
% A=randn(500,3); 
% [B,kflags]=madout(A);
% ndims(B)==ndims(A)
% sum(kflags)==size(B,1)
% 
% % Comparison to TRIMIT
% madout('demo1');
%
% Last modified by olwalbert-at-princeton.edu, 01/29/2026

defval('kappa',3.5)

if ~isstr(A)
  % Calculate along the column's
  dim=1;
  % MATLAB'S median absolute deviation
  a=mad(A,1,dim);
  % Sample median
  aa=median(A,dim);
  modZ=0.6745*(A-aa)./a;
  % Keep only if all parameters are within the bounds
  kflags=logical(prod(abs(modZ)<=kappa,2));
  B=A(kflags,:);
  C=A(~kflags,:);
elseif strcmp(A,'demo1')
  A=randn(10000,3);
  [B,kflags]=madout(A);
  [C,tflags]=trimit(A,99.);
  FA=1;
  clf;
  subplot(131)
  histogram(A(:,1),'FaceAlpha',1)
  xline(minmax(A(:,2)),'--',{'A'},'HandleVisibility','off')
  hold on
  histogram(B(:,1),'FaceAlpha',FA)
  xline(minmax(B(:,1)),'--',{'M'},'HandleVisibility','off')
  histogram(C(:,1),'FaceAlpha',FA)
  xline(minmax(C(:,1)),'--',{'T'},'HandleVisibility','off')
  legend(sprintf('ALL , N=%i,m=%0.2f,sd=%0.2f',size(A,1),mean(A(:,1)),std(A(:,1))),...
         sprintf('MAD , N=%i,m=%0.2f,sd=%0.2f',size(B,1),mean(B(:,1)),std(B(:,1))),...
         sprintf('TRIM, N=%i,m=%0.2f,sd=%0.2f',size(C,1),mean(C(:,1)),std(C(:,1))),...
         location='northoutside')
  subplot(132)
  histogram(A(:,2),'FaceAlpha',1)
  xline(minmax(A(:,2)),'--',{'A'},'HandleVisibility','off')
  hold on
  histogram(B(:,2),'FaceAlpha',FA)
  xline(minmax(B(:,2)),'--',{'M'},'HandleVisibility','off')
  histogram(C(:,2),'FaceAlpha',FA)
  xline(minmax(C(:,2)),'--',{'T'},'HandleVisibility','off')
  legend(sprintf('ALL , N=%i,m=%0.2f,sd=%0.2f',size(A,1),mean(A(:,2)),std(A(:,2))),...
         sprintf('MAD , N=%i,m=%0.2f,sd=%0.2f',size(B,1),mean(B(:,2)),std(B(:,2))),...
         sprintf('TRIM, N=%i,m=%0.2f,sd=%0.2f',size(C,1),mean(C(:,2)),std(C(:,2))),...
         location='northoutside')
  subplot(133)
  histogram(A(:,3),'FaceAlpha',1)
  xline(minmax(A(:,3)),'--',{'A'},'HandleVisibility','off')
  hold on
  histogram(B(:,3),'FaceAlpha',FA)
  xline(minmax(B(:,3)),'--',{'M'},'HandleVisibility','off')
  histogram(C(:,3),'FaceAlpha',FA)
  xline(minmax(C(:,3)),'--',{'T'},'HandleVisibility','off')
  legend(sprintf('ALL , N=%i,m=%0.2f,sd=%0.2f',size(A,1),mean(A(:,3)),std(A(:,3))),...
         sprintf('MAD , N=%i,m=%0.2f,sd=%0.2f',size(B,1),mean(B(:,3)),std(B(:,3))),...
         sprintf('TRIM, N=%i,m=%0.2f,sd=%0.2f',size(C,1),mean(C(:,3)),std(C(:,3))),...
         location='northoutside')
end
