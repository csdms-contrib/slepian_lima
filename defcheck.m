function flag=defcheck(A,tol,meth)
% FLAG=DEFCHECK(A,tol,meth)
%
% Checks if a matrix is (semi-)positive definite, (semi-)negative definite, or
% indefinite by either comparing sign of eigenvalues (within a tolerance if
% semi-), or through Sylvester's criterion, and returns a flag.
%
% INPUT:
%
% A        The matrix
% tol      The negative of the tolerance exponent
% meth     1 Check the sign of the eigenvalues
%          2 Check the sign of the leading principal minors; Sylvester's criterion 
%
% OUTPUT:
%
% flag     0 Positive definite
%          1 Semi-positive definite
%          2 Negative definite
%          3 Semi-negative definite
%          4 Indefinite
%          5 Not Hermitian
% 
% EXAMPLES:
%
% % To check if positive or semi-positive definite,
% flg=defcheck(A,12,randi(2));diferm(flg>1)
% % Comparing methods
% diferm(defcheck(A,[],1)-defcheck(A,[],2))
%
% Last modified by olwalbert-at-princeton.edu, 03/28/2025

defval('tol',15)
defval('meth',1)

% First confirm that the matrix is Hermitian
lastwarn('')
hermcheck(A)
[warnMsg, warnId] = lastwarn;
if ~isempty(warnMsg)
  % Not Hermitian
  flag=5;
else
  switch meth
    case 1
      % Check the eigenvalues
      B=eig(A);
    case 2
      % Check the leading principal minors
      for ind=1:size(A,1)
        B(ind)=det(A(1:ind,1:ind));
      end
  end
  if all(B>0)
    flag=0;
  elseif all(B>=-10^(-tol))
    flag=1;
  elseif all(B<0)
    flag=2;
  elseif all(B<=10^(-tol))
    flag=3;
  else
    flag=4;
  end
end
