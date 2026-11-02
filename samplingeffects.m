function samplingeffects(upad,ifptrn)
%
% Figure studying the spatial window and spatial kernel (w/o) padding
%
% INPUTS
% 
% upad      logical scalar -  padding
% ifptrn    logical vector - [random checker sweep bound1]
% 
% SEE ALSO
%
% OPTCOV, where the model enters the stage
%
% EXAMPLES
%
% samplingeffects(0,[1 0 0 0])
%
% Last modified by olwalbert-at-princeton.edu, 02/10/2026

% Starting scenario will be the random deletion tapers from the covgammiosl
% demos for infill and growing domain experiments

% Which patterns to display, all options by default
defval('ifptrn',[1 1 1 1 0])

% How many sampling patterns to consider 
numptrns=sum(ifptrn);

% How many "experiments" - infill, growing, reduced
numexps=3;

% percent observed for infill and growing domain
mscl=2/3;

% preallocate space for the tapers
tpr=cell(numptrns,3,numexps);

% Random missingness
if ifptrn(1)
  fromcovdemo=0;
  try 
    % infill
    if fromcovdemo
      demo7vars2=load('covgammiosl_demo7_calcs_20-Jan-2026_random.mat');
      tpr{1,2,1}=demo7vars2.tapersv{1};
      tpr{1,2,2}=demo7vars2.tapersv{2};
      tpr{1,2,3}=demo7vars2.tapersv{3};
      ny{1,2}=demo7vars2.ny;
      dy{1,2}=demo7vars2.dy;
    else
      %ny{1,1}=[64 128 256];
      ny{1,2}=[16 32 64];
      dy{1,2}=[4 2 1];
      for nnd=1:numel(dy{1,2})
        p=[];p.NyNx=[ny{1,2}(nnd) ny{1,2}(nnd)];p.dydx=[dy{1,2}(nnd) dy{1,2}(nnd)];
        p.mask='random';
        [~,I]=muckit(ones(p.NyNx),p,2/3);                                         
        tpr{1,2,nnd}=I;                                                           
      end
    end
    obp{1,2}=round([sum(tpr{1,2,1}(:)) sum(tpr{1,2,2}(:)) sum(tpr{1,2,3}(:))]./...
                   ny{1,2}.^2,2)*100;

    % growing 
    ny{1,1}=ny{1,2};
    dy{1,1}=[1 1 1];
    for nnd=1:numel(dy{1,1})
      p=[];p.NyNx=[ny{1,1}(nnd) ny{1,1}(nnd)];p.dydx=[dy{1,1}(nnd) dy{1,1}(nnd)];
      p.mask='random';
      [~,I]=muckit(ones(p.NyNx),p,2/3);
      tpr{1,1,nnd}=I;
    end
    obp{1,1}=round([sum(tpr{1,1,1}(:)) sum(tpr{1,1,2}(:)) sum(tpr{1,1,3}(:))]./...
                   ny{1,1}.^2,2)*100;

    % reduced
    ny{1,3}=[ny{1,1}(1) ny{1,1}(1) ny{1,1}(1)];
    dy{1,3}=[1 1 1];
    p=[];p.NyNx=[ny{1,3}(nnd) ny{1,3}(nnd)];p.dydx=[dy{1,3}(nnd) dy{1,3}(nnd)];
    p.mask='random';
    for nnd=1:numel(dy{1,3})
      [~,I]=muckit(ones(p.NyNx),p,3/5+3/20*(nnd-1));
      [~,I]=muckit(ones(p.NyNx),p,1/50+5/20*(nnd-1));
      tpr{1,3,nnd}=I;
    end
    obp{1,3}=round([sum(tpr{1,3,1}(:)) sum(tpr{1,3,2}(:)) sum(tpr{1,3,3}(:))]./...
                   ny{1,3}.^2,2)*100;
  catch
    warning('could not load mask for random missingness')
    keyboard
  end
end

% Checkered missingness
if ifptrn(2)
  try
    ckscl=14;
    % infill
    ny{2,1}=ny{1,1};
    dy{2,1}=dy{1,1};
    for nnd=1:numel(dy{2,1})
      p=[];p.NyNx=[ny{2,1}(nnd) ny{2,1}(nnd)];p.dydx=[dy{2,1}(nnd) dy{2,1}(nnd)];
      p.mask='checker';
      [~,I]=mixit(ones(p.NyNx),p,ckscl);                                         
      tpr{2,1,nnd}=I;                                                           
    end
    obp{2,1}=round([sum(tpr{2,1,1}(:)) sum(tpr{2,1,2}(:)) sum(tpr{2,1,3}(:))]./...
                   ny{2,1}.^2,2)*100;

    % growing 
    ny{2,2}=ny{1,2};
    dy{2,2}=dy{1,2};
    for nnd=1:numel(dy{2,2})
      p=[];p.NyNx=[ny{2,2}(nnd) ny{2,2}(nnd)];p.dydx=[dy{2,2}(nnd) dy{2,2}(nnd)];
      p.mask='checker';
      [~,I]=mixit(ones(p.NyNx),p,ckscl);                                         
      tpr{2,2,nnd}=I;                                                           
    end               
    obp{2,2}=round([sum(tpr{2,2,1}(:)) sum(tpr{2,2,2}(:)) sum(tpr{2,2,3}(:))]./...
                   ny{2,2}.^2,2)*100;

    % reduced
    ny{2,3}=[ny{2,1}(3) ny{2,1}(3) ny{2,1}(3)];
    dy{2,3}=[1 1 1];
    p=[];p.NyNx=[ny{2,3}(3) ny{2,3}(3)];p.dydx=[dy{2,3}(3) dy{2,3}(3)];
    p.mask='checker';
    for nnd=1:numel(dy{1,3})
      [~,I]=mixit(ones(p.NyNx),p,ckscl);
      % Fill in some of the zeros as asymmetric quadrants
      if nnd==2
        I(1:floor(ny{2,3}(3)/2),1:floor(ny{2,3}(3)/2))=deal(1);
      elseif nnd==3
        I(1:floor(ny{2,3}(3)/2),1:floor(ny{2,3}(3)/2))=deal(1);
        I(floor(ny{2,3}(3)/2):end,floor(ny{2,3}(3)/2):end)=deal(1);
      end
      tpr{2,3,nnd}=I;
    end
    obp{2,3}=round([sum(tpr{2,3,1}(:)) sum(tpr{2,3,2}(:)) sum(tpr{2,3,3}(:))]./...
                   ny{2,3}.^2,2)*100;
  catch
    warning('could not load mask for checkering -- skipping')
    keyboard
    ifptrn(4)=0;
    numptrns=sum(ifptrn);
  end
end

% Structured missingness
if ifptrn(5)
  try
    [~,tpr{3,:}]=maskit();
  catch
    ifptrn(2)=0;
    warning('could not load mask for structured sweeps -- skipping')
    numptrns=sum(ifptrn);
  end
end

% Bounded missingness -- make this 4 later
if ifptrn(3)
  try
    % infill
    ny{3,1}=ny{1,1};
    dy{3,1}=dy{1,1};
    for nnd=1:numel(dy{1,2})
      p=[];p.NyNx=[ny{3,1}(nnd) ny{3,1}(nnd)];p.dydx=[dy{3,1}(nnd) dy{3,1}(nnd)];
      p.mask='newjersey';                                                          
      [~,I]=maskit(ones(p.NyNx),p,2/3);                                         
      tpr{3,1,nnd}=I;                                                           
    end
    obp{3,1}=round([sum(tpr{3,1,1}(:)) sum(tpr{3,1,2}(:)) sum(tpr{3,1,3}(:))]./...
                   ny{3,1}.^2,2)*100;

    % growing 
    ny{3,2}=ny{1,2};
    dy{3,2}=dy{1,2};
    for nnd=1:numel(dy{3,2})
      p=[];p.NyNx=[ny{3,2}(nnd) ny{3,2}(nnd)];p.dydx=[dy{3,2}(nnd) dy{3,2}(nnd)];
      p.mask='newjersey';                                                          
      [~,I]=maskit(ones(p.NyNx),p,2/3);                                         
      tpr{3,2,nnd}=I;                                                           
    end               
    obp{3,2}=round([sum(tpr{3,2,1}(:)) sum(tpr{3,2,2}(:)) sum(tpr{3,2,3}(:))]./...
                   ny{3,2}.^2,2)*100;

    % reduced
    ny{3,3}=[ny{3,1}(3) ny{3,1}(3) ny{3,1}(3)];
    dy{3,3}=[1 1 1];
    p=[];p.NyNx=[ny{3,3}(nnd) ny{3,3}(nnd)];p.dydx=[dy{3,3}(nnd) dy{3,3}(nnd)];
    p.mask='newjersey';                                                          
    for nnd=1:numel(dy{3,3})
      [~,I]=maskit(ones(p.NyNx),p,2/5+7/20*(nnd-1));
      tpr{3,3,nnd}=I;                                                           
    end               
    obp{3,3}=round([sum(tpr{3,3,1}(:)) sum(tpr{3,3,2}(:)) sum(tpr{3,3,3}(:))]./...
                   ny{3,3}.^2,2)*100;
  catch
    warning('could not load mask for a boundary -- skipping')
    keyboard
    ifptrn(3)=0;
    numptrns=sum(ifptrn);
  end
end

% Bounded missingness -- exterior
if ifptrn(4)
  try
    % infill
    ny{4,1}=ny{1,1};
    dy{4,1}=dy{1,1};
    for nnd=1:numel(dy{1,1})
      tpr{4,1,nnd}=~tpr{3,1,nnd};
    end
    obp{4,1}=round([sum(tpr{4,1,1}(:)) sum(tpr{4,1,2}(:)) sum(tpr{4,1,3}(:))]./...
                   ny{4,1}.^2,2)*100;

    % growing 
    ny{4,2}=ny{1,2};
    dy{4,2}=dy{1,2};
    for nnd=1:numel(dy{1,2})
      tpr{4,2,nnd}=~tpr{3,2,nnd};
    end
    obp{4,2}=round([sum(tpr{4,2,1}(:)) sum(tpr{4,2,2}(:)) sum(tpr{4,2,3}(:))]./...
                   ny{4,2}.^2,2)*100;

    % reduced
    ny{4,3}=[ny{4,1}(3) ny{4,1}(3) ny{4,1}(3)];
    dy{4,3}=[1 1 1];
    p=[];p.NyNx=[ny{4,3}(nnd) ny{4,3}(nnd)];p.dydx=[dy{4,3}(nnd) dy{4,3}(nnd)];
    p.mask='newjersey';                                                          
    for nnd=1:numel(dy{4,3})
      [~,I]=maskit(ones(p.NyNx),p,4/5-7/20*(nnd-1));
      tpr{4,3,nnd}=~I;                                                           
    end               
    obp{4,3}=round([sum(tpr{4,3,1}(:)) sum(tpr{4,3,2}(:)) sum(tpr{4,3,3}(:))]./...
                   ny{4,3}.^2,2)*100;
  catch
    warning('could not load mask for a boundary -- skipping')
    keyboard
    ifptrn(2)=0;
    numptrns=sum(ifptrn);
  end
end

% Calculate the spectral window
for mnd=1:numptrns
  mimas{mnd} =[];
  mimaps{mnd}=[];
  for jnd=1:numexps
    for ind=1:numel(dy{jnd})
      ftx0=fftshift(fft2(tpr{mnd,jnd,ind}));
      ftph0=imag(ftx0);
      ftph{mnd,jnd,ind}=ftph0;
      ftx0=decibel(abs(ftx0).^2);
      % Decibel will produce -Inf if an element is 0; replace these with the
      % next smallest value
      ftx0(isinf(ftx0))=NaN;
     %% ftx0(isnan(ftx0))=min(ftx0,[],'all');
      ftx{mnd,jnd,ind}=ftx0;
      mimas{mnd}=[mimas{mnd} minmax(ftx0)];

      % And again, but this time with padding
      ftx0p=fftshift(fft2(tpr{mnd,jnd,ind},2^nextpow2(ny{mnd,jnd}(ind)+1),...
                                           2^nextpow2(ny{mnd,jnd}(ind)+1)));
      ftph0p=imag(ftx0p);
      ftphp{mnd,jnd,ind}=ftph0p;
      ftx0p=decibel(abs(ftx0p).^2);
      % Decibel will produce -Inf if an element is 0; replace these with the
      % next smallest value
      ftx0p(isinf(ftx0p))=NaN;
     %% ftx0p(isnan(ftx0p))=min(ftx0p,[],'all');
      ftxp{mnd,jnd,ind}=ftx0p;
      mimaps{mnd}=[mimaps{mnd} minmax(ftx0p)];
    end
  end
end

% With or without padding
if upad 
  ftp=ftxp; ftphs=ftphp; 
else 
  ftp=ftx;  ftphs=ftph; 
end

% Create the figure
figure(50);                                                                    
clf                                                                          
spr=200*3*numptrns;
spc=950*1;
set(gcf,'InnerPosition',[1 1 spc spr])                                       
[ah,ha]=krijetem(subnum(numptrns*numexps,6)); 

axs=1:size(ha,2);
xnds=axs(~~mod(axs,2));
knds=axs(~mod(axs,2));
for mnd=1:numptrns
  for jnd=1:numexps
    for ind=1:size(dy{1,1},2)
      % Display the taper in grid space
      rnd=ind+(size(dy{1,1},2))*(jnd-1)+(size(dy{1,1},2))*(numexps)*(mnd-1);
      xnd=xnds(rnd);
      knd=knds(rnd);
      axes(ah(xnd))
      imagesc(tpr{mnd,jnd,ind})
      colormap(gca,gray(2))
      caxis([0 1])
      hold on

      % If we had a model in mind, we could show the decorrelation length scale 
      % (the correlation is 1/3 of the variance at a distance of pi*rho, unitless)
      %  % dc(ind,1:2)=circ(pi*th1(3)/dy{mnd,jnd}(ind),[],...
      %  %                 [pi*th1(3)/dy{mnd,jnd}(ind) ny{jnd}(ind)-pi*th1(3)/dy{mnd,jnd}(ind)]);
      %  % set(dc(ind,1),'color','b'); delete(dc(ind,2))
      % However, we will keep models out of it for now

      if jnd==1
        % bold the Nx
        tti(mnd,jnd,ind)=title(sprintf(...
                               '%s%s$_%i$=%i, $%sx$_%i$=%i, %s%i%s%s$',...
                               '\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad',...
                               '\textbf{Nx}',ind,ny{mnd,jnd}(ind),...
                               '$\Delta$',ind,dy{mnd,jnd}(ind),...
                               'Obs=',obp{mnd,jnd}(ind),'\textbf{\%}'),...
                               '$w(\mathbf{x})$',...
                         'Interpreter','latex');
      elseif jnd==2
        % bold the dx
        tti(mnd,jnd,ind)=title(sprintf(...
                               '%s Nx$_%i$=%i, $%s$_%i$=%i, %s%i%s%s$',...
                               '\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad',...
                               ind,ny{mnd,jnd}(ind),...
                               '$\mathbf{\Delta}$\textbf{x}',ind,dy{mnd,jnd}(ind),...
                               'Obs=',obp{mnd,jnd}(ind),'\textbf{\%}'),...
                               '$w(\mathbf{x})$',...
                         'Interpreter','latex');
      elseif jnd==3
        % bold the Obs
        tti(mnd,jnd,ind)=title(sprintf(...
                               '%s Nx$_%i$=%i, $%sx$_%i$=%i, %s%i%s%s$',...
                               '\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad',...
                               ind,ny{mnd,jnd}(ind),...
                               '$\Delta$',ind,dy{mnd,jnd}(ind),...
                               '\textbf{Obs=}',obp{mnd,jnd}(ind),'\textbf{\%}'),...
                               '$w(\mathbf{x})$',...
                         'Interpreter','latex');
      end
      axis image
      shrink(ah(xnd),1.05,1.05)

      if jnd<3
        % growing domain or infill experiment
        xtis=[1 ny{mnd,jnd}(1:ind)];
        xtils={'1','Nx_1','Nx_2','Nx_3'};
        xtil=xtils(1:ind+1);
        xtil={'','','',''};
        xtil{1}=xtils{1}; xtil{ind+1}=xtils{ind+1};
      else
        % the missingness reduction experiment only has one grid size
        xtis=[1 ny{mnd,jnd}(1)];
        xtils={'1','Nx_1'};
        xtil={''};
        xtil{1}=xtils{1}; xtil{2}=xtils{2};
      end

      %set(ah(xnd),'XTick',[1 ny{mnd,jnd}(ind)],'YTick',[1 ny{mnd,jnd}(ind)])
      set(ah(xnd),'XTick',xtis,'YTick',xtis,'XTickLabel',xtil,'YTickLabel',xtil,...
                  'XTickLabelRotation',0)
      longticks
      xlabel('$x$','Interpreter','latex')
      ylabel('$y$','Interpreter','latex')

      % Display the taper in wavenumber space
      axes(ah(knd))
      pnd=1;
      if pnd==1
           % square modulus pf the FT of the taper
           imagesc(ftp{mnd,jnd,ind})
         %  [tah{mnd,jnd,ind},cx{mnd,jnd,ind}]=imagefnan(...
         %       [],[],ftp{mnd,jnd,ind},[],'parula')
           tti2(ind)=title('$|w(\mathbf{k})|^2$','Interpreter','latex');
      else
           % Phase of the FT of the taper
           imagesc(ftphs{mnd,jnd,ind})
      end
      longticks
      axis image
      shrink(ah(knd),1.05,1.05)
      if upad==1
        %set(ah(knd),'XTick',[1 ny{mnd,jnd}(ind)],'YTick',[1 ny{mnd,jnd}(ind)])
        set(ah(knd),'XTick',xtis,'YTick',xtis,'XTickLabel',xtil,'YTickLabel',xtil,...
                  'XTickLabelRotation',0)
      else
        %set(ah(knd),'XTick',[1 size(ftxp{ind},1)],'YTick',[1 size(ftxp{ind},2)])
        set(ah(knd),'XTick',xtis,'YTick',xtis,'XTickLabel',xtil,'YTickLabel',xtil,...
                  'XTickLabelRotation',0)
      end

      xlabel('$k_x$','Interpreter','latex')
      ylabel('$k_y$','Interpreter','latex')
      set(ah(knd),'YAxisLocation','right')

      moveh([ah(xnd) ah(knd)],-0.01*(3-4*(ind-1)))
      moveh(ah(knd),-0.02)
    end
  end
end
% Go back and adjust the color axes
for mnd=1:numptrns
  for jnd=1:numexps
    for ind=1:size(dy{1,1},2)
      % Display the taper in grid space
      rnd=ind+(size(dy{1,1},2))*(jnd-1)+(size(dy{1,1},2))*(numexps)*(mnd-1);
      knd=knds(rnd);
      axes(ah(knd))
      %colormap(gca,parula)%nebula)%sky)
      if upad
         caxis(minmax(mimaps{mnd}))
      else
         caxis(minmax(mimas{mnd}))
      end
     % caxis([-80 -20])
    end
  end
end


% Let the panels breathe in the vertical
if numptrns>3
  movev(ah(1:6*numexps),0.04)
  movev(ah(6*numexps+1:6*numexps*2),0.015)
  movev(ah(end-6*numexps*2+1:end-6*numexps),-0.015)
  movev(ah(end-6*numexps+1:end),-0.04)
end
% Let the experiments breathe in the vertical
for mnd=1:numptrns
  for jnd=1:numexps
    if jnd==2
      movev(ah(6*jnd+6*(mnd-1)*numexps-5:6*jnd+6*(mnd-1)*numexps),-0.003)
    elseif jnd==3
      movev(ah(6*jnd+6*(mnd-1)*numexps-5:6*jnd+6*(mnd-1)*numexps),-0.005)
    end
  end
end


for mnd=1:numptrns
  % Set the colorbars
  cbx{mnd}=colorbar(ah((numexps-1)*mnd*6+1),'Location','southoutside');
  cbx{mnd}.Position(1)=0.15;
  cbx{mnd}.Position(3)=0.3;
  cbx{mnd}.Position(4)=0.005;
  cbx{mnd}.Label.String=sprintf('spatial grid space: %s','$w(\mathbf{x})$');
  cbx{mnd}.Label.Interpreter='latex';
  %cbx{mnd}.Label.FontSize=8.5;
  cbx{mnd}.AxisLocation='in';
  cbx{mnd}.XTick=[0 1];
  cbx{mnd}.Label.Position(2)=1.3;
  
  cbk{mnd}=colorbar(ah(numexps*mnd*6),'Location','southoutside');
  cbk{mnd}.Position(1)=0.6;
  cbk{mnd}.Position(3)=0.3;
  cbk{mnd}.Position(4)=0.005;
  if upad 
    cbk{mnd}.XTick=ceil(minmax(mimaps{mnd}));
  else
    cbk{mnd}.XTick=ceil(minmax(mimas{mnd}));
  end
  if pnd==1
    cbk{mnd}.Label.String=sprintf('wavenumber space: %s',...
      '$10\mathrm{log}10(|w(\mathbf{k})|^2/\mathrm{max}(|w(\mathbf{k})|^2))$');
    cbk{mnd}.Label.Interpreter='latex';
    %cbk{mnd}.Label.FontSize=8.5;
  else
    cbk{mnd}.Label.String='wavenumber space: phase';
  end
  cbk{mnd}.AxisLocation='in';
  cbk{mnd}.Label.Position(2)=1.3;
  if mnd==1
    cbx{mnd}.Position(2)=0.745;
    cbk{mnd}.Position(2)=0.745;
  elseif mnd==2
    cbx{mnd}.Position(2)=0.515;
    cbk{mnd}.Position(2)=0.515;
  elseif mnd==3
    cbx{mnd}.Position(2)=0.275;
    cbk{mnd}.Position(2)=0.275;
  elseif mnd==4
    cbx{mnd}.Position(2)=0.04;
    cbk{mnd}.Position(2)=0.04;
  end
end

keyboard
% Pause to confirm the figure is nice before saving
saveas(gcf,...
  sprintf('samplingeffects_%s_%i.eps',date,upad),'epsc')
