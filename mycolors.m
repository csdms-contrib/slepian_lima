function clrs=mycolors(num)
% FUNCTION CLRS=MYCOLORS(NUM)
%
% Colors I like
%
% Last modified by olwalbert-at-princeton.edu, 03/17/2026

defval('num',1)

switch num
  case 1
    % Color friendly palette
    clrs=["#1B9E77","#D95F02","#7570B3"];
    clrs=sscanf(char(clrs),'#%2x%2x%2x',[3,size(clrs,2)]).'/255;
  case 2
    % Blues
    clrs=["#0033FF","#3399FF","#33ccFF"];
    clrs=sscanf(char(clrs),'#%2x%2x%2x',[3,size(clrs,2)]).'/255;
  case 3
    clrs=["#0033FF","#D95F02"];
    clrs=sscanf(char(clrs),'#%2x%2x%2x',[3,size(clrs,2)]).'/255;
  case 4
    % Diverging colormap
    clrs2=mycolors(3);
    clrs2(2,1)=1;
    clrs=interp1([0;0.5;1],...
                 [clrs2(2,:);[1 1 1];clrs2(1,:)],...
                 linspace(0,1,256),...
                 'pchip');
  case 5
    % the Venus region color scheme
    gld=[255 215 0]./255;
    grn=[0 150 120]./255;
    blu=[0 0 128]./255;
    pnk=[232 93 131]./255;
    clrs=[pnk; grn; gld; blu];
  end
end
