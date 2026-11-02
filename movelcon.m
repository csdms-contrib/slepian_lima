function varargout=movelcon(fax,amt)
% [new,old]=MOVELCON(fax,amt)
%
% Move legend icons by some amount, defaulted by icon object type.
% 
% INPUT:
%
% fax      Legend icon handles
% amt      Amount of move (+ right towards text, - left away from text) 
%
% OUTPUT:
%
% new      The new XData position(s)
% old      The old XData position(s)
%
% SEE ALSO:
%
% MOVEH, MOVEV
%
% Last modified by olwalbert-at-princeton.edu, 01/20/2026 


legicms=findobj(fax,'Type','line');
% Whether the object is a line or marker is evidenced by the size of X.Data
for knd=1:size(legicms,1)
  oldpos{knd,:}=legicms(knd).XData;
  if size(legicms(knd).XData,1)==1
    defval('amt',[0.3 0 0])
    legicms(knd).XData=amt(1);
  else
    defval('amt',[0 0.2 0.4])
    legicms(knd).XData(1)=amt(2);
    legicms(knd).XData(2)=amt(3);
  end
  newpos{knd,:}=legicms(knd).XData;
end

% Provide desired output
varns={newpos,oldpos};
varargout=varns(1:nargout);
