function bath2synth
% function BATH2SYNTH
% Make a pretty opening figure. Not spending too much time on this now. All the
% machinery to explore many data, model, and presentation options are in
% pltGEBCO.m.
%
% EXAMPLES
%
% bath2synth
% 
% Last modified by olwalbert-at-princeton.edu, 09/06/2026

% Subset of isotropic like section with pretty track composition
mask00=[37146       32450];
maskMN=[37746       33050];
% The associated lon,lat coords
sz=[ -25.2271   45.2063;  -22.7271   47.7063];
mclons=[-25.2271 -22.721];
mclats=[45.2063 47.7063];

% Read and parse the flag values attribute which tells us the source of the gridded
% value: whether it is from a direct measurement or not; see the                
% GEBCO_Grid_Document for more details                                          
pth='~/Documents/OceanFloor';
gtidatt_flags=ncreadatt(sprintf('%s/gebco_2024_tid/GEBCO_2024_TID.nc',pth),'tid','flag_values');
% Read the flag meaning attribute associated with the flag values               
gtidatt_flagm=ncreadatt(sprintf('%s/gebco_2024_tid/GEBCO_2024_TID.nc',pth),'tid','flag_meanings');
% Split at the whitespace                                                       
gtidatt_flagm=split(gtidatt_flagm);                                             
% Item 16 does not parse as easily                                              
gtidatt_flagm{19}=gtidatt_flagm{18};gtidatt_flagm{18}=gtidatt_flagm{17};        
gtidatt_flagm{17}=gtidatt_flagm{16}(18:end);                                    
gtidatt_flagm{16}=gtidatt_flagm{16}(1:17);                                      
% Land is flag value 0                                                          
landflag=0;                                                                     
% Direct measurement flag values are 10--17: Singlebeam, Multibeam, Seismic,    
% Isolated_sounding, ENC_sounding, Lidar, Optical, Combined                     
dirflags=[10:17];                                                               
                                                                                
% Read the rest of the grid's metadata                                          
% The grid TID                                                                  
gtid=ncread('gebco_2024_tid/GEBCO_2024_TID.nc','tid')';                         
% The grid longitude values                                                     
glon=ncread('gebco_2024_tid/GEBCO_2024_TID.nc','lon');                          
% The grid latitudue values                                                     
glat=ncread('gebco_2024_tid/GEBCO_2024_TID.nc','lat');           
% Elevation is relative to mean sea level
gelev=ncread('GEBCO_2024_sub_ice_topo.nc','elevation')';

% Direct measurements only
gdir_mask=ismember(gtid,dirflags);
% Land only
landmask=gtid==landflag;

maskflgsa=gtid(mask00(2):maskMN(2), mask00(1):maskMN(1));
maskflgs=unique(maskflgsa);
for ind=1:numel(maskflgs)
  % get the names of the tids
  masktgs(ind)=gtidatt_flagm(maskflgs(ind)==gtidatt_flags);
end
% For the masked area, the direct measurements include: multibeam;
% ...and interpolated methods: predicted_altimetry, bathymetric_sounding
% What percentage of the grid in the pretty mask area is from direct measurement?
% -> 49.6%
maskprc=sum(ismember(maskflgsa,dirflags),"all")./prod(size(maskflgsa));

% Select for our data
[~,lonidx(1)]=min(abs(glon-sz(1,1))); [~,lonidx(2)]=min(abs(glon-sz(2,1)));   
[~,latidx(1)]=min(abs(glat-sz(1,2))); [~,latidx(2)]=min(abs(glat-sz(2,2)));   
lm=int16(landmask(latidx(1):latidx(2), lonidx(1):lonidx(2)));                 
gm=int16(gdir_mask(latidx(1):latidx(2),lonidx(1):lonidx(2)));                 
% immasks=int16(2.*gm.*int16(~lm)+lm);                                          
mlons=glon(lonidx(1):lonidx(2));                                              
mlats=glat(latidx(1):latidx(2));                                              
% Decimal percent direct measurement                                          
dirprc=sum(gm,"all")/prod(size(gm));       
% All data within the region of interest
sgelevf=gelev(latidx(1):latidx(2),lonidx(1):lonidx(2));                      

% Our grid parameters
p=[];                                                                       
p.NyNx=size(sgelevf);                                                        
% spacing in km for 15 second resolution
p.dydx=[deg2km(min(abs(mlons(2:end)-mlons(1:end-1))))...                    
        deg2km(min(abs(mlats(2:end)-mlats(1:end-1))))];                     
p.blurs=-1;                                                                 
% The full grid taper
pf=p;pf.taper=1;                                                                  
% The masked grid taper
pm=p;pm.taper=~~gm;                                                                  
% The anti-masked grid taper
ngm=~gm;
pa=p;pa.taper=~~ngm;                                                                  

% Decimate? Look at more than just simulations?
xtra=1;
xxtra=0;
% Use existing estimates?
eest=1;

if ~xxtra
  % Our data, full field
  Hxf=double(sgelevf(:));
  [~,~,c,~,~,Z]=planefit(v2s(Hxf,p));
  meanf=mean(Hxf);
  Hxdf=Hxf-Z(:);
  
  % Our data, the masked field
  sgelevm=double(sgelevf);
  sgelevm(gm==0)=deal(NaN);
  Hxm=sgelevm(:);
  meanm=nanmean(Hxm);
  Hxdm=Hxm-Z(:);
  meandm=nanmean(Hxdm);
  Hxdm=Hxdm-meandm;
  
  % Our data, the anti-masked field
  sgeleva=double(sgelevf);
  sgeleva(ngm==0)=deal(NaN);
  Hxa=sgeleva(:);
  meana=nanmean(Hxa);
  Hxda=Hxa-Z(:);
  meanda=nanmean(Hxda);
  Hxda=Hxda-meanda;
else
  decm=6;
  % Our data, full field
  Hxf=double(sgelevf(1:decm:end,1:decm:end));
  decp=pf;decp.NyNx=size(Hxf);decp.dydx=decp.dydx.*decm;decp.taper=1;
  Hxf=Hxf(:);
  [~,~,c,~,~,Z]=planefit(v2s(Hxf,decp));
  meanf=mean(Hxf);
  Hxdf=Hxf-Z(:);
  
  % Our data, the masked field
  sgelevm=double(sgelevf(1:decm:end,1:decm:end));
  gm=gm(1:decm:end,1:decm:end);
  sgelevm(gm==0)=deal(NaN);
  Hxm=sgelevm(:);
  meanm=nanmean(Hxm);
  Hxdm=Hxm-Z(:);
  meandm=nanmean(Hxdm);
  Hxdm=Hxdm-meandm;
  
  % Our data, the anti-masked field
  sgeleva=double(sgelevf(1:decm:end,1:decm:end));
  ngm=ngm(1:decm:end,1:decm:end);
  sgeleva(ngm==0)=deal(NaN);
  Hxa=sgeleva(:);
  meana=nanmean(Hxa);
  Hxda=Hxa-Z(:);
  meanda=nanmean(Hxda);
  Hxda=Hxda-meanda;
end


keyboard
% Estimate for the full field
if eest
  if ~xxtra
    thhatf=[38871.3184 1.4604 2.8418];
  else
    % for the decimated field
     thhatf=[5.194844196202718e+04 1.224303720012647 4.569864984072007];
  end
else
  if ~xxtra
     thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf,[],pf); end    
  else
    % try estimating for a decimated field to be consistent with the naive
    % estimator
   % decm=6;
   % decHx  =v2s(Hxdf,pf);
   % decHx  =decHx(1:decm:end,1:decm:end);
   % decntpr=pm.taper(1:decm:end,1:decm:end);
   % decatpr=pa.taper(1:decm:end,1:decm:end);
    decHx=Hxdf;
    decp.taper=1;
    thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(decHx(:),[],decp); end    
    thhatf=thhatu.*scl;                                                          
  end
end
% Synthetic for the full field
pf.blurs=Inf;
Hxsf=v2s(simulosl(thhatf,pf),pf);

% Estimate for the mask field
if eest
  if ~xxtra
    thhatm=[45452.2537 0.9608 4.3221];
  else
    % for the decimated field
    thhatm=[4.749040183989936e+04  0.974752399983638  4.337976803174395];
  end
else
  if ~xxtra
    thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf-meandm,[],pm); end    
  else
    decp.taper=pm.taper(1:decm:end,1:decm:end);
    thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(decHx(:),[],decp); end    
    thhatm=thhatu.*scl;                                                          
  end
end
% Synthetic for the mask field
pm.blurs=Inf;
try
  Hxsmq=v2s(simulosl(thhatm,pf),pf);
  Hxsm=Hxsmq.*pm.taper;
catch
  % circulant embedding might fail
  try
    pmd=pf;pmd.NyNx=pmd.NyNx.*2;
    Hxsm=v2s(simulosl(thhatm,pmd),pmd);
    % quarter it?
    Hxsmq=quarter(v2s(Hxsm,pmd));
    Hxsm=Hxsmq.*pm.taper;
  catch
    keyboard
  end
end
% Estimate for the anti field
if eest
  if ~xxtra
    thhata=[47762.6886 1.4508 4.8812];
  else
    % for the decimated field
    thhata=[4.789713027645076e+04  1.466062491425266 4.646138066763448];
  end
else
  if ~xxtra
    thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf-meanda,[],pa); end    
  else
    decp.taper=pa.taper(1:decm:end,1:decm:end);
    thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(decHx(:),[],decp); end    
    thhata=thhatu.*scl;                                                          
  end
end
% Synthetic for the anti field
pa.blurs=Inf;
try
  Hxsaq=v2s(simulosl(thhata,pf),pf);
  Hxsa=Hxsaq.*pa.taper;
catch
  % circulant embedding might fail
  try
    pad=pf;pad.NyNx=pad.NyNx.*2;
    Hxsa=v2s(simulosl(thhata,pad),pad);
    % quarter it?
    Hxsaq=quarter(v2s(Hxsa,pad));
    Hxsa=Hxsaq.*pa.taper;
  catch
    keyboard
  end
end

% Common color range
% [cmap,~,ziro]=sergeicol;
% cmap=cmap(1:ziro,:);

%cax=prctile([Hx ; HxS],[0 100])+km;
% Lighten up by subtracting right
%cax=cax+[0 -500];
if xtra & ~xxtra
  % On the final xtra axes, we will plot 68.3% confidence intervals as horizontal
  % error bars for the three cases: direct, indirect, and merged parameter
  % estimates. This will require us to calculate the parameter uncertainties 
  covm=covthosl(thhatm,pm,1,[1 1 1]);
  cova=covthosl(thhata,pa,1,[1 1 1]);
  covf=covthosl(thhatf,pf,1,[1 1 1]);
elseif xxtra
  % for the decimated field which we made the estimate for
  decp.taper=pm.taper(1:decm:end,1:decm:end);
  covm=covthosl(thhatm,decp,2,[1 1 1]);
  decp.taper=pa.taper(1:decm:end,1:decm:end);
  cova=covthosl(thhata,decp,2,[1 1 1]);
  decp.taper=1;
  covf=covthosl(thhatf,decp,2,[1 1 1]);
end

% Make the figure
figure(11);clf; clear cb xcb
if xxtra
  set(gcf,'InnerPosition',[1 1 900 900])
  ah=krijetem(subnum(5,3));                                                   
  for ind=1:15
    ah(ind).Position(4)=0.1577;
  end
elseif xtra
  set(gcf,'InnerPosition',[1 1 900 900])
  ah=krijetem(subnum(4,3));                                                   
else
  set(gcf,'InnerPosition',[1 1 900 600])
  ah=krijetem(subnum(2,3));                                                   
end
% Data, synthetics
Hxsfm=Hxsf(:)+Z(:);%+c;
Hxsmm=Hxsm(:)+Z(:)+meandm;%+c+meandm;
Hxsmqm=Hxsmq(:)+Z(:)+meandm;
Hxsmm(gm==0)=deal(NaN);
Hxsam=Hxsa(:)+Z(:)+meanda;%+c+meanda;
Hxsaqm=Hxsaq(:)+Z(:)+meanda;
Hxsam(ngm==0)=deal(NaN);
dat={Hxm,Hxa,Hxf,Hxsmm,Hxsam,Hxsfm,Hxsmqm,Hxsaqm};
labs={'direct','indirect','merged'};
cma=abyss(2);
cma=cma(1,:);
cma=[1 1 1];
cmap='kelicol';
for ind=1:6
  axes(ah(ind));                                                                
  if ind<4
    cax(ind,:)=[1.15 0.75].*prctile([dat{ind}; dat{ind+3}(:)],[5 95]);
    [~,ocax{ind}]=imagefnan([mclons(1) mclats(1)],[mclons(2) mclats(2)],...           
                  v2s(dat{ind},p),cmap,cax(ind,:),cma,1);
    ti(ind)=title(sprintf('%s',labs{ind}));
    movev(ti(ind),0.1)
    mima{ind}=minmax([dat{ind}; dat{ind+3}(:)]);
    [cb{ind},xcb{ind}]=addcb('hor',mima{ind},mima{ind},cmap);
    cb{ind}.XLabel.String=('elevation [m]');
    ah(ind).XTickLabel='';
  else
    imagefnan([mclons(1) mclats(1)],[mclons(2) mclats(2)],...           
               v2s(dat{ind},p),cmap,cax(ind-3,:),cma,1) 
    xlabel(sprintf('longitude (%s)', char(176)))
  end
  axes(ah(ind))
  longticks; box on
  if ind==1||ind==4
    ylabel(sprintf('latitude (%s)', char(176)))                                 
    ah(ind).XTick=unique(round(ah(ind).XTick));
  elseif ind==2||ind==5
    ah(ind).YTickLabel='';
  else
    ah(ind).YAxisLocation='right';
  end
end
for ind=1:3
  cb{ind}.Position(1)=ah(ind).Position(1);
  cb{ind}.Position(3)=ah(ind).Position(3);
  if xtra
    cb{ind}.Position(2)=0.65;
  else
    cb{ind}.Position(2)=0.49;
  end
  cb{ind}.XLim=round(mima{ind});
  cb{ind}.XTick=round(mima{ind});
  cb{ind}.XTickLabel={'',''};
  cb{ind}.XTickLabel=round(mima{ind});
  cb{ind}.XAxisLocation='top';
  cb{ind}.XLabel.Position(2)=3.3;
end
if xxtra
  for ind=4:6
    axes(ah(ind))
    xlabel(sprintf('longitude (%s)', char(176)))
    axes(ah(ind))
    longticks; box on
  end
elseif xtra
  for ind=4:8
    if ind<7
      ah(ind).XTickLabel='';
      ah(ind).XLabel.String='';
    else
      axes(ah(ind))
      imagefnan([mclons(1) mclats(1)],[mclons(2) mclats(2)],...           
                 v2s(dat{ind},p),cmap,cax(ind-6,:),cma,1) 
      xlabel(sprintf('longitude (%s)', char(176)))
      axes(ah(ind))
      longticks; box on
      if ind==7
        ylabel(sprintf('latitude (%s)', char(176)))                                 
        ah(ind).XTick=unique(round(ah(ind).XTick));
      elseif ind==8
        ah(ind).YTickLabel='';
      %else
      %  ah(ind).YAxisLocation='right';
      end
    end
  end
end
if xtra | xxtra
  % tick labels
  ytratks=[0.7 0.8 0.9];
  ytratklbs={'mer','ind','dir'};

  % MLE with uncertainties for the three sampling scenarios
  axes(ah(9))
  hold on
  longticks; box on
  % s2
  clrs={[1 0.74 0.38],[1 0.65 0.25],[0.70 1 0.87]};
  clrs=mycolors(1);
  es2m=errorbar(thhatm(1),0.9,sqrt(covm(1,1)),'horizontal',...
                '^','MarkerFaceColor',clrs(1,:),'Color',clrs(1,:));
  es2a=errorbar(thhata(1),0.8,sqrt(cova(1,1)),'horizontal',...
                'v','MarkerFaceColor',clrs(2,:),'Color',clrs(2,:));
  es2f=errorbar(thhatf(1),0.7,sqrt(covf(1,1)),'horizontal',...
                'diamond','MarkerFaceColor',clrs(3,:),'Color',clrs(3,:));
  ah(9).XLim=[31000 57000];
  ah(9).YLim=[0.65 0.95];
  ah(9).YTick=ytratks;
  ah(9).YTickLabel=ytratklbs;
  ah(9).XLabel.String='variance \sigma^2 [m^2]';
  %ah(9).XLabel.Position(1)=ah(9).XLabel.Position(1);
  ah(9).XLabel.Position(1)=mean(ah(9).XLim);
  ah(9).XLabel.Position(2)=0.48;
  ah(9).FontSize=8.5;
  ah(9).XLabel.FontSize=9.5;

  axes(ah(10))
  hold on
  longticks; box on
  % nu
  enum=errorbar(thhatm(2),0.9,sqrt(covm(2,2)),'horizontal',...
                '^','MarkerFaceColor',clrs(1,:),'Color',clrs(1,:));
  enua=errorbar(thhata(2),0.8,sqrt(cova(2,2)),'horizontal',...
                'v','MarkerFaceColor',clrs(2,:),'Color',clrs(2,:));
  enuf=errorbar(thhatf(2),0.7,sqrt(covf(2,2)),'horizontal',...
                'diamond','MarkerFaceColor',clrs(3,:),'Color',clrs(3,:));
  ah(10).YLim=[0.65 0.95];
  ah(10).XLim=[0.75 1.72];
  ah(10).YTick=ytratks;
  ah(10).YTickLabel=ytratklbs;
  ah(10).XLabel.String='smoothness \nu';
  ah(10).XLabel.Position(2)=0.48;
  ah(10).FontSize=8.5;
  ah(10).XLabel.FontSize=9.5;

  axes(ah(11))
  hold on
  longticks; box on
  % rh
  erhm=errorbar(thhatm(3),0.9,sqrt(covm(3,3)),'horizontal',...
                '^','MarkerFaceColor',clrs(1,:),'Color',clrs(1,:));
  erha=errorbar(thhata(3),0.8,sqrt(cova(3,3)),'horizontal',...
                'v','MarkerFaceColor',clrs(2,:),'Color',clrs(2,:));
  erhf=errorbar(thhatf(3),0.7,sqrt(covf(3,3)),'horizontal',...
                'diamond','MarkerFaceColor',clrs(3,:),'Color',clrs(3,:));
  ah(11).YLim=[0.65 0.95];
  ah(11).XLim=[2.5 5.9];
  ah(11).YTick=ytratks;
  ah(11).YTickLabel=ytratklbs;
  ah(11).XLabel.String='range \rho [km]';
  ah(11).XLabel.Position(2)=0.48;
  ah(11).FontSize=8.5;
  ah(11).XLabel.FontSize=9.5;

  % Enlarge the data maps
  shrink(ah([1:8]),0.75,0.75)
  movev(ah([4:12]),-0.06)
  movev(cb{1},0.045)
  movev(cb{2},0.045)
  movev(cb{3},0.045)
  movev(ah([7:12]),-0.03)

  % shrink 9 through 11, move 10 and 11
  ah(9).Position(4)=0.04;
  %ah(9).Position(2)=0.458;
  ah(9).Position(2)=0.383;

  ah(10).Position(4)=0.04;
  ah(10).Position(1)=ah(9).Position(1);
  ah(10).Position(2)=0.293;

  ah(11).Position(4)=0.04;
  ah(11).Position(1)=ah(9).Position(1);
  ah(11).Position(2)=0.205;
end
if xxtra
  loadsaved=1;
  % Addition of naive estimators for s2 and correlation length:
  ntpr=double(pm.taper(:));
  ntpr(ntpr==0)=NaN;
  ntpr=reshape(ntpr,size(pm.taper));
  atpr=double(pa.taper(:));
  atpr(atpr==0)=NaN;
  atpr=reshape(atpr,size(pa.taper));
  % interior, exterior, merged
  %nvs2=[nanvar((Hxdf-meandm).*ntpr(:)) nanvar((Hxdf-meanda).*atpr(:)) nanvar(Hxdf)];
  %Ns=[nansum(ntpr(:)) nansum(atpr(:)) prod(pf.NyNx)];
  %nvs2unc=2.*nvs2.^2./(Ns-1); % variance of s^2
  % for the full rectangle, dy==dx, which makes this simpler
  % We will have to decimate the dataset for SPPDIST; 601 by 601 is asking too
  % much
  decm=6;
  decHx=v2s(Hxdf,pf);
  decHx=decHx(1:decm:end,1:decm:end);
  decntpr=ntpr(1:decm:end,1:decm:end);
  decatpr=atpr(1:decm:end,1:decm:end);
  decp=pf;decp.NyNx=size(decHx);decp.dydx=decp.dydx.*decm;
  siz=decp.NyNx;
  % repeat the naive variance calculation for the decimated field
  nvs2=[nanvar((decHx(:)-meandm).*decntpr(:))...
        nanvar((decHx(:)-meanda).*decatpr(:))...
        nanvar(decHx(:))];
  Ns=[nansum(decntpr(:)) nansum(decatpr(:)) prod(decp.NyNx)];
  nvs2unc=2.*nvs2.^2./(Ns-1); % variance of s^2
  if loadsaved
   % load('bath2synth_naivcov.mat')
   % some pretty, abstract data art
   % imagesc(datforcovsme)
   % imagesc(datforcovsin)
   % imagesc(datforcovsex)
   load('bath2synth_taketwo.mat')
  else
    % Generate vectors of data that correspond to all lag-pair distances, then
    % calculate their correlation coefficients
    %
    xys=1:prod(siz);
    sxps=nan(decp.NyNx(1),decp.NyNx(2),prod(siz));
    for knd=1:prod(siz)
      sxp=sspdist(knd,xys,siz)';
      % [~,dis(knd,:),djs(knd,:)]=sspdist(knd,xys,siz);
      sxps(:,:,knd)=reshape(single(sxp),siz);
    end
    usxps=unique(sxps(:));
    %
    % datforcovsme=nan(numel(usxps),prod(siz));
    % datforcovsin=nan(numel(usxps),prod(siz));
    % datforcovsex=nan(numel(usxps),prod(siz));
    %
    sxmes=nan(numel(usxps),1);
    sxins=nan(numel(usxps),1);
    sxexs=nan(numel(usxps),1);
    %
    sx2mes=nan(numel(usxps),1);
    sx2ins=nan(numel(usxps),1);
    sx2exs=nan(numel(usxps),1);
    %
    stdmes=nan(numel(usxps),1);
    stdins=nan(numel(usxps),1);
    stdexs=nan(numel(usxps),1);
    %
    for ind=1:numel(usxps)
      sxme=0;sx2me=0;Nlagsme=0;
      sxin=0;sx2in=0;Nlagsin=0;
      sxex=0;sx2ex=0;Nlagsex=0;
      for jnd=1:prod(siz)
        laginds=sxps(:,:,jnd)==usxps(ind);
        Nlagsme=Nlagsme+sum(laginds(:));
        Nlagsin=Nlagsin+nansum(ntpr(laginds));
        Nlagsex=Nlagsex+nansum(atpr(laginds));
        % visual check
         % keyboard
         % clf;imagesc(laginds);
         % title(sprintf('lag %0.2f,comparison %i',usxps(ind),jnd))
        %
        % data pertaining to lag-pairs are stored down the column; the matrix
        % element that lag-pairs pertain to is organized along the row
        prdsme=decHx(laginds).*decHx(jnd);
        sxme=nansum(prdsme)+sxme;
        sx2me=nansum(prdsme.^2)+sx2me;
        %
        prdsin=(decHx(laginds)-meandm).*decntpr(laginds).*decHx(jnd);
        sxin=nansum(prdsin)+sxin;
        sx2in=nansum(prdsin.^2)+sx2in;
        %
        prdsex=(decHx(laginds)-meanda).*decatpr(laginds).*decHx(jnd);
        sxex=nansum(prdsex)+sxex;
        sx2ex=nansum(prdsex.^2)+sx2ex;
        %
        % datforcovsme(ind,jnd)=nanmean(decHx(laginds).*decHx(jnd));
        % datforcovsin(ind,jnd)=nanmean((decHx(laginds)-meandm).*ntpr(laginds).*decHx(jnd));
        % datforcovsex(ind,jnd)=nanmean((decHx(laginds)-meanda).*atpr(laginds).*decHx(jnd));
      end
      sxmes(ind)= sxme./Nlagsme;
      sx2mes(ind)=sx2me./Nlagsme;
      stdmes(ind)=sqrt((sx2mes(ind)-sxmes(ind).^2)./Nlagsme);
      %
      sxins(ind)=  sxin./Nlagsin;
      sx2ins(ind)=sx2in./Nlagsin;
      stdins(ind)=sqrt((sx2ins(ind)-sxins(ind).^2)./Nlagsin);
      %
      sxexs(ind)=  sxex./Nlagsex;
      sx2exs(ind)=sx2ex./Nlagsex;
      stdexs(ind)=sqrt((sx2exs(ind)-sxexs(ind).^2)./Nlagsex);
    end
    save bath2synth_taketwo sxmes sx2mes stdmes sxins sx2ins stdins sxexs sx2exs  stdexs usxps
  end
  
  % approximate the range parameter using the 1/(3pi) rule-of-thumb
  approxrhmeind=find((sxmes./sxmes(1)-1/3)<0,1,'first');
  approxrhme=usxps(approxrhmeind).*decp.dydx(1)./pi;
  approxrhinind=find((sxins./sxins(1)-1/3)<0,1,'first');
  approxrhin=usxps(approxrhinind).*decp.dydx(1)./pi;
  approxrhexind=find((sxexs./sxexs(1)-1/3)<0,1,'first');
  approxrhex=usxps(approxrhexind).*decp.dydx(1)./pi;
  %
  % the naive range estimator
  nvrh=[approxrhin approxrhex approxrhme];
  %
  % This likely will not come back
  %
  %  % average covariance for lag-distance across all pairs
  %  naivcovme=nanmean(datforcovsme,2);
  %  naivcovin=nanmean(datforcovsin,2);
  %  naivcovex=nanmean(datforcovsex,2);
  %  % correlation for each dataset
  %  ncorrme=naivcovme./naivcovme(1);
  %  ncorrin=naivcovin./naivcovin(1);
  %  ncorrex=naivcovex./naivcovex(1);
  %  % standard deviation 
  %  nerrcorrme=nanstd(datforcovsme'./naivcovme(1))';
  %  nerrcorrin=nanstd(datforcovsin'./naivcovin(1))';
  %  nerrcorrex=nanstd(datforcovsex'./naivcovex(1))';
  % % decs=20;
  % % keyboard
  % % figure()
  % % errorbar(lagds(1:decs:end),ncorrme(1:decs:end),nerrcorrme(1:decs:end))
  % % errorbar(lagds(1:decs:end),ncorrin(1:decs:end),nerrcorrin(1:decs:end))
  % % errorbar(lagds(1:decs:end),ncorrex(1:decs:end),nerrcorrex(1:decs:end))
  %  % find the lag-distance where the correlation drops to 1/3
  % % approxrhmeind=abs(ncorrme-1/3)<0.01;
  %  approxrhmeind=find((ncorrme-1/3)<0,1,'first');
  %  approxrhme=usxps(approxrhmeind).*decp.dydx(1)./pi;
  %  approxrhinind=find((ncorrin-1/3)<0,1,'first');
  %  approxrhin=usxps(approxrhinind).*decp.dydx(1)./pi;
  %  approxrhexind=find((ncorrex-1/3)<0,1,'first');
  %  approxrhex=usxps(approxrhexind).*decp.dydx(1)./pi;
  %
  % intercept if plus 1 s.d.?
  upncorrme=(sxmes(:,1)+stdmes(:,1))./(sxmes(1)+stdmes(1));
  upapproxrhmeind=find((upncorrme-1/3)<0,1,'first');
  upapproxrhme=usxps(upapproxrhmeind).*decp.dydx(1)./pi;
  upncorrin=(sxins(:,1)+stdins(:,1))./(sxins(1)+stdins(1));
  upapproxrhinind=find((upncorrin-1/3)<0,1,'first');
  upapproxrhin=usxps(upapproxrhinind).*decp.dydx(1)./pi;
  upncorrex=(sxexs(:,1)+stdexs(:,1))./(sxexs(1)+stdexs(1));
  upapproxrhexind=find((upncorrex-1/3)<0,1,'first');
  upapproxrhex=usxps(upapproxrhexind).*decp.dydx(1)./pi;
  % intercept if minus 1 s.d.?
  loncorrme=(sxmes(:,1)-stdmes(:,1))./(sxmes(1)-stdmes(1));
  loapproxrhmeind=find((loncorrme-1/3)<0,1,'first');
  loapproxrhme=usxps(loapproxrhmeind).*decp.dydx(1)./pi;
  loncorrin=(sxins(:,1)-stdins(:,1))./(sxins(1)-stdins(1));
  loapproxrhinind=find((loncorrin-1/3)<0,1,'first');
  loapproxrhin=usxps(loapproxrhinind).*decp.dydx(1)./pi;
  loncorrex=(sxexs(:,1)-stdexs(:,1))./(sxexs(1)-stdexs(1));
  loapproxrhexind=find((loncorrex-1/3)<0,1,'first');
  loapproxrhex=usxps(loapproxrhexind).*decp.dydx(1)./pi;
  %
  % the standard deviation of the naive rho estimator
  nvrhunc=[mean([abs(upapproxrhin-approxrhin) abs(loapproxrhin-approxrhin)])...
           mean([abs(upapproxrhex-approxrhex) abs(loapproxrhex-approxrhex)])...
           mean([abs(upapproxrhme-approxrhme) abs(loapproxrhme-approxrhme)])];
  %  
  %  % alternatively, fit an exponential curve to the average and the +/- 1s.d. covariances
  %  % and find the lag-distance that corresponds to these errors so that we can
  %  % propagate errors into our reported naive estimate for the range parameter
  %  
  % plot the empirical correlation
  axes(ah(8))
  box on;longticks
  hold on
  %decs=186;
  decs=1;
  lagds=usxps.*decp.dydx(1);
  nclrs=clrs+0.149;
  ncvp(2)=errorbar(lagds(1:decs:end),sxins(1:decs:end,1)./sxins(1),stdins(1:decs:end,1)./sxins(1),'^',...
               'Color',nclrs(1,:),'DisplayName','dir');
  ncvp(3)=errorbar(lagds(1:decs:end),sxexs(1:decs:end,1)./sxexs(1),stdexs(1:decs:end,1)./sxexs(1),'v',...
               'Color',nclrs(2,:),'DisplayName','ind');
  ncvp(1)=errorbar(lagds(1:decs:end),sxmes(1:decs:end,1)./sxmes(1),stdmes(1:decs:end,1)./sxmes(1),'diamond',...
               'Color',nclrs(3,:),'DisplayName','mer');
  for nnd=1:3;ncvp(nnd).MarkerSize=4;ncvp(nnd).CapSize=0;ncvp(nnd).LineStyle='-';end
  lg=legend();
  lg.Location='northeast';
  nyl(1)=yline(1/3,'b-','\approx\pi\rho','HandleVisibility','off','FontSize',8.5);
  nyl(1).Label='1/3';
  nyl(1).LabelHorizontalAlignment='left';
  nxl(1)=xline(approxrhme*pi,'-','Color',nclrs(3,:),...
               'HandleVisibility','off','FontSize',9);
  nxl(2)=xline(approxrhin*pi,'-','Color',nclrs(1,:),...
               'Label','\approx\pi\rho','HandleVisibility','off','FontSize',9);
  nxl(3)=xline(approxrhex*pi,'-','Color',nclrs(2,:),...
               'HandleVisibility','off','FontSize',9);
  nxl(2).LabelVerticalAlignment='top';
  nxl(2).LabelHorizontalAlignment='left';
  nyl(1).Color=[0.3 0.3 1];
  xlabel('lag-distance r, [km]')
  ylabel('$\tilde{C}(r)/\tilde{C}(0)$','Interpreter','latex')
  ylim([-0.05 1.05])
  xlim([-1 50])
  grid on
  ah(8).FontSize=8.5;
  ah(8).XLabel.FontSize=9.5;
  ah(8).YLabel.FontSize=9.5;
  ah(8).XColor=[0.3 0.3 0.3];
  ah(8).YColor=[0.3 0.3 0.3];
  lg.EdgeColor=[0.3 0.3 0.3];
  lg.TextColor=[0.3 0.3 0.3];

  axes(ah(12))
  hold on
  longticks; box on
  % ns2
  nes2m=errorbar(nvs2(1),0.9,sqrt(nvs2unc(1)),'horizontal',...
              '^','Color',nclrs(1,:));%,'Color',clrs(1,:));
  nes2a=errorbar(nvs2(2),0.8,sqrt(nvs2unc(2)),'horizontal',...
              'v','Color',nclrs(2,:));%,'Color',clrs(2,:));
  nes2f=errorbar(nvs2(3),0.7,sqrt(nvs2unc(3)),'horizontal',...
              'diamond','Color',nclrs(3,:));%,'Color',clrs(3,:));
  % ah(12).XLim=[31000 57000];
  ah(12).YLim=[0.65 0.95];
  ah(12).XLim=[4.87 5.21].*1e4;
  ah(12).YTick=ytratks;
  ah(12).YTickLabel=ytratklbs;
  ah(12).XLabel.String='naive variance [m^2]';
  ah(12).XLabel.Position(1)=5.02e4;%mean(ah(12).XLim);
  ah(12).XLabel.Position(2)=0.48;
  ah(12).FontSize=8.5;
  ah(12).XLabel.FontSize=9.5;
  ah(12).XColor=[0.3 0.3 0.3];
  ah(12).YColor=[0.3 0.3 0.3];

  axes(ah(13))
  hold on
  longticks; box on
  ntnum(1)=text(1,0.89,'?','Color',nclrs(1,:));
  ntnum(2)=text(1,0.8,'?','Color', nclrs(2,:));
  ntnum(3)=text(1,0.71,'?','Color',nclrs(3,:));
  for nnd=1:3;ntnum(nnd).FontSize=9;end
  ah(13).YLim=[0.65 0.95];
  ah(13).XLim=[0.75 1.25];
  ah(13).XTick='';
  ah(13).YTick=ytratks;
  ah(13).YTickLabel=ytratklbs;
  ah(13).XLabel.String='naive smoothness';
  ah(13).XLabel.Position(2)=0.48;
  ah(13).XLabel.Position(2)=0.55;
  ah(13).FontSize=8.5;
  ah(13).XLabel.FontSize=9.5;
  ah(13).XColor=[0.3 0.3 0.3];
  ah(13).YColor=[0.3 0.3 0.3];

  axes(ah(14))
  hold on                                                                     
  longticks; box on                                                           
  % nrh                                                                       
  nerhm=errorbar(nvrh(1),0.9,nvrhunc(1),'horizontal',...                
              '^','Color',nclrs(1,:));%,'Color',clrs(1,:));             
  nerha=errorbar(nvrh(2),0.8,nvrhunc(2),'horizontal',...                
              'v','Color',nclrs(2,:));%,'Color',clrs(2,:));             
  nerhf=errorbar(nvrh(3),0.7,nvrhunc(3),'horizontal',...                
              'diamond','Color',nclrs(3,:));%,'Color',clrs(3,:));       
 % ah(14).XLim=[0.65 0.95];                                                  
  ah(14).YLim=[0.65 0.95];                                                     
  ah(14).XLim=[4.5 6.5];
  ah(14).YTick=ytratks;                                                        
  ah(14).YTickLabel=ytratklbs;                                                 
  ah(14).XLabel.String='naive range [km]';                             
  %ah(9).XLabel.Position(1)=ah(9).XLabel.Position(1);                         
  ah(14).XLabel.Position(1)=mean(ah(14).XLim);                                  
  ah(14).XLabel.Position(2)=0.48;                                              
  ah(14).FontSize=8.5;                                                         
  ah(14).XLabel.FontSize=9.5;                   
  ah(14).XColor=[0.3 0.3 0.3];
  ah(14).YColor=[0.3 0.3 0.3];

  % Shrink and move 12 through 14 approx to 7, remove 7 and 15
  ah(12).Position(4)=0.04;
  ah(12).Position(1)=ah(7).Position(1);
  ah(12).Position(2)=0.383;
  ah(13).Position(4)=0.04;
  ah(13).Position(1)=ah(12).Position(1);
  ah(13).Position(2)=0.293;
  ah(14).Position(4)=0.04;
  ah(14).Position(1)=ah(12).Position(1);
  ah(14).Position(2)=0.205;
  
  delete([ah(7) ah(15)])

  % last bit of fitting
  movev(ah([8:14]),-0.01)
  movev([cb{1} cb{2} cb{3}],0.175)
  moveh(ah([2:3,5:6,8:11]),-0.02)
  moveh([cb{2} cb{3}],-0.02)
  moveh(ah([3,6,9:11]),-0.02)
  moveh([cb{3}],-0.02)
  ah(3).YTickLabel='';
  ah(6).YTickLabel='';
  ti(1).FontWeight='normal';
  ti(2).FontWeight='normal';
  ti(3).FontWeight='normal';
  movev(ti([1:3]),0.2)
  movev([ah([1:6]) ah([8:14])],-0.1)
  movev([ah([1:6]) ah(8)],-0.03)
  movev(ti,0.2)
  movev(ah(8:14),0.04)
  moveh(ah(12:14),0.037)
  ah(9).YAxisLocation='right';
  ah(10).YAxisLocation='right';
  ah(11).YAxisLocation='right';
  %
  %ah(8).Position(1)=0.389;
  ah(8).Position(1)=0.405;
  ah(8).Position(2)=0.135;
  ah(8).Position(3)=0.201;
  ah(8).Position(4)=0.22;
elseif xtra & ~xxtra
  delete(ah(12))
  % last bit of fitting
  movev(ah([7:11]),-0.01)
  movev([cb{1} cb{2} cb{3}],0.175)
  moveh(ah([2:3,5:6,8:11]),-0.02)
  moveh([cb{2} cb{3}],-0.02)
  moveh(ah([3,6,9:11]),-0.02)
  moveh([cb{3}],-0.02)
  ah(3).YTickLabel='';
  ah(6).YTickLabel='';
  ti(1).FontWeight='normal';
  ti(2).FontWeight='normal';
  ti(3).FontWeight='normal';
  movev(ti([1:3]),0.2)
  movev(ah([1:11]),-0.1)
  if xxtra
    ah(11).XLim=[3.8 6.4];
    ah(14).XLim=[3.8 6.4];
    ah(9).XLim=[4.2 6.1]*1e4;
    ah(12).XLim=[4.2 6.1]*1e4;
    ah(9).XLabel.Position(1)=mean(ah(9).XLim);
    ah(11).XLabel.Position(1)=mean(ah(11).XLim);
    ah(14).XLabel.Position(1)=mean(ah(14).XLim);
  end
end
keyboard

%Tx=gettaper(pf,'cosine',0.1);
%pmt=pm;pmt.taper=pm.taper.*Tx;pat=pa;pat.taper=pa.taper.*Tx;pft=pf;pft.taper=Tx;
%pmt.kiso=NaN;pat.kiso=NaN;pft.kiso=NaN;
%ma=mlechipsdosl(tospec(Hxf.*Tx(:),pmt),thhatm,[1 1 1],pmt);
%aa=mlechipsdosl(tospec(Hxf.*Tx(:),pat),thhata,[1 1 1],pat);
%fa=mlechipsdosl(tospec(Hxf.*Tx(:),pft),thhatf,[1 1 1],pft);

figna=figdisp([],sprintf('%s.eps',date),[],1);
%ti(1)=title(sprintf('GEBCO 2024 bathymetry, %0.2f%% direct',dirprc*100));   
