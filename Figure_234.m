 %% FIGURE 2
  figure(102)
clf;
 for type = 3:5
 load(['DistFromOpt_' num2str(type) '.mat'])
 
for sess = 1:3 
    for tri = 1:4

        DT_SP_root = Trial_Length{sess,tri};
        

HH = histcounts(DT_SP_root,1:0.25:5,'Normalization','probability');

subplot(3,4,sess+(type-3)*4)
plot(1:0.25:4.75,smoothdata(HH,'movmean',2),'LineWidth',2,'Color',[1-tri/4 0 tri/4])
ylim([0 0.4])
xlabel('Relative Trial Length')
ylabel('Probability')
switch type 
    case 3
title({'Build-Up',['RTL Session ' num2str(sess)]})

    case 4
    title({'Location Update',['RTL Session ' num2str(sess)]})
    case 5
     title({'Barrier Update',['RTL Session ' num2str(sess)]})   
end
hold on


    end

end


  subplot(3,4,4+(type-3)*4)
  for sess=1:3
plot(Direct_P(sess,:),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
ylim([0 0.45])
hold on
  end
xlabel('Trial Group')
xticklabels({'1','2-11','12-21','22-31'})
ylabel('Prob of Direct Run')
switch type
    case 3
title({'Build-Up','Probability RTL<1.5'})
    case 4
title({'Location Update','Probability RTL<1.5'})
    case 5
title({'Barrier Update','Probability RTL<1.5'})
end


hold on   

 end

 subplot(3,4,1)
 legend('Trial 1','Trials 2-11','Trials 12-21','Trials 22-31')
subplot (3,4,4)
legend('Session 1','Session 2','Session 3','Location','best')

%% FIGURE 3
figure(101)
clf;
for type = 3:5
load(['DistFromOpt_' num2str(type) '.mat'])


 
ll = 0;
for sess = 1:3


for tri = 1:4
ll = ll+1;
subplot(3,4,sess+(type-3)*4)
hold on
 plot(All_means(:,ll),'LineWidth',2,'Color',[1-tri/4 0 tri/4])
xlabel('Node choices')
ylabel('Mean DFOP')
ylim([0 3])
xlim([0 50]) %only for the scaled paths
%title(['Distance from ideal path (Wrong First Turn) Session 1-4 (All nodes) ' num2str(sess)])


if(tri==1)


subplot(3,4,4+(type-3)*4)
hold on
plot(All_means(:,ll),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
xlabel('Nodes choices')
ylabel('Mean DFOP')
ylim([0 3])
xlim([0 50]) %only for the scaled paths
switch type
    case 3
title({'Build-up',['First Trial of Each Session']})
    case 4
title({'Location Update',['First Trial of Each Session']})
    case 5
title({'Barrier Update',['First Trial of Each Session']})
end
grid minor
end


end

switch type 
    case 3
title({' Build-up ' , ['Session ', num2str(sess)]})
    case 4
title({' Location Update ', ['Session ', num2str(sess)]})
    case 5
title({ ' Barrier Update ',['Session ', num2str(sess)]})
end






end
 subplot(3,4,1)
 legend('Trial 1','Trials 2-11','Trials 12-21','Trials 22-31')
   subplot(3,4,4)
 legend('1st Trial, S 1','1st Trial, S 2','1st Trial, S 3')
 
 
end


%% FIGURE 4 EXAMPLE

  figure(103)
 clf;
 
 for type = 3:3

load(['DistFromOpt_' num2str(type) '.mat'])
 

 ll = 0;
 for sess = 1:3
 for tri = 1:4
     ll = ll+1;
  if(sess > 1 || tri ~=2)
  continue
  end
     Grand_mean = All_means(:,ll);   
     
     
 x_label = (1:size(All_means,1))';
% k=0;



y = Grand_mean;%    smoothdata(Grand_mean,'movmean',1);
   
F_Fit = @(x,xdata)x(1)*1/(exp(-2*log(x(3)/x(2))/(1-(x(2)/x(3)).^2))-exp(-2*log(x(3)/x(2))/((x(3)/x(2)).^2-1)))*(exp(-xdata.^2/(2*x(2).^2))-exp(-xdata.^2/(2*x(3).^2)));

x0 = [1.5 20 5];
lb = [0 0 0];
ub = [Inf Inf Inf];
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
[x] = lsqcurvefit(F_Fit,x0,x_label,y,lb,ub,options);


subplot(1,1,type-2)
plot(y,'Color',[112,128,144]/256,'LineWidth',2)
hold on
plot(F_Fit(x,x_label),'Color',[255,0,255]/256,'LineWidth',2,'LineStyle','--')
ylim([0 2])
xlim([0 45])
legend('Experiment','Best Fit')
switch type
    case 3
    title({'Parametric Fit'})%,['Session', num2str(sess)]})
    case 4
    title({'Location Update',['Session', num2str(sess)]})
    case 5
    title({'Barrier Update',['Session', num2str(sess)]})  
end
hold on
%pause()
 end
xlabel('Node choices')
ylabel('Mean DFOP') 
set(gca,'FontSize',14)
 end

 end
 %% FIGURE 4 PARA
   figure(104)
 clf;
for type = 3:5
load(['DistFromOpt_' num2str(type) '.mat'])
 

 for pp=1:3
    subplot(3,3,pp+(type-3)*3)
    FP = reshape(Fits_Para(:,pp),[],3);
    for sess = 1:3
     plot((FP(:,sess)),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
     hold on
    end
     xlabel('Trial Group')
     xticklabels({'1','2-11','12-21','22-31'})
ylabel('Fit Value')


switch type
    case 3
        
switch pp
    case 1
        title({'Build-up','Peak'})
        ylim([1 2])
    case 2
        title({'Build-up', 'Descending Length'})
        ylim([7 14])
    case 3
         title({'Build_up','Ascending Length'})
         ylim([1.5 2.5])
end
    case 4
switch pp
    case 1
        title({'Location Update','Peak'})
        ylim([1 2])
    case 2
        title({'Location Update', 'Descending Length'})
        ylim([7 14])
    case 3
         title({'Location Update','Ascending Length'})
         ylim([1.5 2.5])
end
    case 5
switch pp
    case 1
        title({'Barrier Update','Peak'})
        ylim([1 2])
    case 2
        title({'Barrier Update', 'Descending Length'})
        ylim([7 14])
    case 3
         title({'Barrier Update','Ascending Length'})
         ylim([1.5 2.5])
end


end

if(pp == 1 && type == 3)
lgd = legend('Session 1','Session 2','Session 3');
lgd.FontSize = 12;

end
 end
 
 
 

 
 
end
