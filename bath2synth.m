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
% Last modified by olwalbert@princeton.edu, 07/05/2026

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
gtidatt_flags=ncreadatt('gebco_2024_tid/GEBCO_2024_TID.nc','tid','flag_values');
% Read the flag meaning attribute associated with the flag values               
gtidatt_flagm=ncreadatt('gebco_2024_tid/GEBCO_2024_TID.nc','tid','flag_meanings');
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

% Use existing estimates?
eest=1;

% Estimate for the full field
if eest
  thhatf=[38871.3184 1.4604 2.8418];
else
  thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf,[],pf); end    
  thhatf=thhatu.*scl;                                                          
end
% Synthetic for the full field
pf.blurs=Inf;
Hxsf=v2s(simulosl(thhatf,pf),pf);

% Estimate for the mask field
if eest
  thhatm=[45452.2537 0.9608 4.3221];
else
  thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf-meandm,[],pm); end    
  thhatm=thhatu.*scl;                                                          
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
  thhata=[47762.6886 1.4508 4.8812];
else
  thhatu=NaN; while isnan(thhatu); [thhatu,~,~,scl]=mleosl(Hxdf-meanda,[],pa); end    
  thhata=thhatu.*scl;                                                          
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
[cmap,~,ziro]=sergeicol;
cmap=cmap(1:ziro,:);

%cax=prctile([Hx ; HxS],[0 100])+km;
% Lighten up by subtracting right
%cax=cax+[0 -500];
xtra=1;
if xtra
  % On the final xtra axes, we will plot 68.3% confidence intervals as horizontal
  % error bars for the three cases: direct, indirect, and merged parameter
  % estimates. This will require us to calculate the parameter uncertainties 
  covm=covthosl(thhatm,pm,1,[1 1 1]);
  cova=covthosl(thhata,pa,1,[1 1 1]);
  covf=covthosl(thhatf,pf,1,[1 1 1]);
end

% Make the figure
figure(11);clf; clear cb xcb
if xtra
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
if xtra
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
  % tick labels
  ytratks=[0.7 0.8 0.9];
  ytratklbs={'mer','ind','dir'};

  axes(ah(9))
  hold on
  longticks; box on
  % s2
  clrs={[1 0.74 0.38],[1 0.65 0.25],[0.70 1 0.87]};
  keyboard
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
  ah(9).XLabel.FontSize=10.5;

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
  ah(10).XLabel.FontSize=10.5;

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
  ah(11).XLabel.FontSize=10.5;

  % Enlarge the data maps
  shrink(ah([1:8]),0.75,0.75)
  movev(ah([4:12]),-0.06)
  movev(cb{1},0.045)
  movev(cb{2},0.045)
  movev(cb{3},0.045)
  movev(ah([7:12]),-0.03)

  % shrink 9 through 11, move 10 and 11, remove 12
  delete(ah(12))
  ah(9).Position(4)=0.04;
  %ah(9).Position(2)=0.458;
  ah(9).Position(2)=0.383;

  ah(10).Position(4)=0.04;
  ah(10).Position(1)=ah(9).Position(1);
  ah(10).Position(2)=0.293;

  ah(11).Position(4)=0.04;
  ah(11).Position(1)=ah(9).Position(1);
  ah(11).Position(2)=0.205;
keyboard

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
end
Tx=gettaper(pf,'cosine',0.1);
pmt=pm;pmt.taper=pm.taper.*Tx;pat=pa;pat.taper=pa.taper.*Tx;pft=pf;pft.taper=Tx;
pmt.kiso=NaN;pat.kiso=NaN;pft.kiso=NaN;
ma=mlechipsdosl(tospec(Hxf.*Tx(:),pmt),thhatm,[1 1 1],pmt);
aa=mlechipsdosl(tospec(Hxf.*Tx(:),pat),thhata,[1 1 1],pat);
fa=mlechipsdosl(tospec(Hxf.*Tx(:),pft),thhatf,[1 1 1],pft);


figna=figdisp([],sprintf('%s.eps',date),[],1);
%ti(1)=title(sprintf('GEBCO 2024 bathymetry, %0.2f%% direct',dirprc*100));   
