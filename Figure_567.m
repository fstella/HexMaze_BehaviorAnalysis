%% FIGURE 5


load DistributionComparison_2.mat

close all
tt = 0;
for type = [3]
tt = tt+1;


for strat = [1]

if strat == 1
load(['parameter_fit_type_' num2str(type) '_LongP_3.mat'])
elseif strat ==2
load(['parameter_fit_type_' num2str(type) '_OptiP_3.mat'])
end


pp = 0;
for PP = [2]
pp = pp+1;
for ss=1:1

%clf;

for tri=1:4
    BB=find(para_search(:,2)==tri & para_search(:,1)==ss & para_search(:,4)==0.1*(PP-1));
    

    
    
    if(ss==1)
    figure(1001)
        subplot(4,2,(tri-1)*2+1)
    plot(para_search(BB,3),para_search(BB,11)+para_search(BB,7),'Color',[1-(strat-1) 0 strat-1])    
    Ind = BB(find((para_search(BB,11)+para_search(BB,7))==min(para_search(BB,11)+para_search(BB,7))));
hold on
    FF = para_search(Ind,3);
    
    scatter(FF,0.05,30,[1-(strat-1) 0 strat-1],'filled','^')
    
    switch tri
       
        case 1
            title('Build-up Session 1 Trial 1')
        case 2
            title('Build-up Session 1 Trials 2-11')
        case 3
            title('Build-up Session 1 Trials 12-21')    
        case 4
            title('Build-up Session 1 Trials 22-31')
    end
    
    ylim([0 1])
    if(tri == 4)
    xlabel('Amount of Foresight')
    end
    ylabel('KS Distance')    
        
    end
    set(gca,'FontSize',14)
    
    if(strat == 1)
    subplot(4,2,(tri-1)*2+2)
    L_Ratio = DistriComp{1,tri};
    DT_SP = DistriComp{2,tri};
    histogram(L_Ratio,'Binedges',[0:0.5:20],'FaceAlpha',0.5,'FaceColor',[218,165,32]/256,'Normalization','cdf')
    hold on
    histogram(DT_SP,'Binedges',[0:0.5:20],'FaceAlpha',0.5,'FaceColor',[112,128,144]/256,'Normalization','cdf')
    legend('Simulated','Experiment')
    end
    xlabel('Relative Trial Length')
    ylabel('Cumulative Prob')
    
    
end

end

end
end
end



%% FIGURE 6


load DistributionComparison.mat

close all
tt = 0;
for type = [3]
tt = tt+1;





pp = 0;
for PP = [3]
pp = pp+1;
for ss=1:1

%clf;

for tri=1:4
    
    

    
    
   

    
    
    
    subplot(4,2,(tri-1)*2+1)
    L_Ratio = DistriComp{3,tri};
    DT_SP = DistriComp{4,tri};
    histogram(L_Ratio,'Binedges',[0:1:10],'FaceAlpha',0.5,'FaceColor',[218,165,32]/256,'Normalization','cdf')
    hold on
    histogram(DT_SP,'Binedges',[0:1:10],'FaceAlpha',0.5,'FaceColor',[112,128,144]/256,'Normalization','cdf')
    legend('Simulated','Experiment')
    
    
        ylabel('Cumulative Prob')
    
    set(gca,'FontSize',14)
    
    
    if(tri==4)
    xlabel('Maximum Stray from Optimal Path','FontSize',16,'FontWeight','bold')
    end
    
    

    
    
    
        switch tri
       
        case 1
            title('Build-up Session 1 Trial 1')
        case 2
            title('Build-up Session 1 Trials 2-11')
        case 3
            title('Build-up Session 1 Trials 12-21')    
        case 4
            title('Build-up Session 1 Trials 22-31')
    end
    
    
end

end

end

end

load DistributionComparison.mat


tt = 0;
for type = [3]
tt = tt+1;





pp = 0;
for PP = [3]
pp = pp+1;
for ss=1:1

%clf;

for tri=1:4
    
    

    
    
   

    
    
    
    subplot(4,2,(tri-1)*2+2)
    L_Ratio = DistriComp{5,tri};
    DT_SP = DistriComp{6,tri};
    histogram(L_Ratio,'Binedges',[0:0.1:1],'FaceAlpha',0.5,'FaceColor',[218,165,32]/256,'Normalization','cdf')
    hold on
    histogram(DT_SP,'Binedges',[0:0.1:1],'FaceAlpha',0.5,'FaceColor',[112,128,144]/256,'Normalization','cdf')
    legend('Simulated','Experiment')
    
       ylabel('Cumulative Prob')
    
    set(gca,'FontSize',14)
    
    if(tri==4)
    xlabel('Fraction of Time in Outer Ring','FontSize',16,'FontWeight','bold')
    end
 
end

end

end

end



%% FIGURE 7

DC = distinguishable_colors(20);

FitV1 = ones(3,4,2,5)*100;
FitV2 = ones(3,4,2,5)*100;
FitS = ones(3,4,2,5)*100;


ForeS=zeros(2,3,4);
ForeI=zeros(2,3,4,4);
Fore_Sig=zeros(13,2,3,4,4);

Border_T = zeros(2,3,4);
Border_M = zeros(2,3,4);


close all
tt = 0;
for type = [3 4 5]
tt = tt+1;


for strat = [1]

if strat == 1
load(['parameter_fit_type_' num2str(type) '_LongP_6.mat'])
elseif strat ==2
load(['parameter_fit_type_' num2str(type) '_OptiP_4.mat'])
end


pp = 0;
for PP = [1]
pp = pp+1;
for ss=1:3

%clf;

for tri=1:4
    BB=find(para_search(:,2)==tri & para_search(:,1)==ss & para_search(:,4)==0.1*(PP-1));
    

    

    
    %Ind = BB(find((para_search(BB,11)+para_search(BB,7))==min(para_search(BB,11)+para_search(BB,7))));
    Ind = BB(find((para_search(BB,7))==min(para_search(BB,7))));

    FF = para_search(Ind,3);
    

    
    
    
    
    if(pp == 1)
    ForeS(strat,ss,tri)=FF(1);
    ForeI(strat,ss,tri,tt)=round(FF(1)/0.2) + 1;
    Fore_Sig(:,strat,ss,tri,tt) = para_search(BB,5);
    
    Border_T(tt,ss,tri) = para_search(BB(1),13);
    Border_M(strat,ss,tri) = para_search(BB(1),14);
    
    
    end
    
    
    
    
    
    
    [FitV1(ss,tri,strat,PP)]=min(para_search(BB,7)+para_search(BB,11));
    [FitV2(ss,tri,strat,PP)]=min(para_search(BB,11));%+para_search(BB,12));
    %[FitV2(ss,tri,strat,PP)]=min(para_search(BB,8)+para_search(BB,12));
    %FitS(ss,tri,strat,PP)= para_search(BB(aa),5);
%     if(ss==1 && tri==1)
%     ForeS(ss,tri)=0;
%     end
    







end


end

if(pp == 1 && tt ==1)
figure(1002)
subplot(1,2,strat)
for ss = 1:3
plot(squeeze(ForeS(strat,ss,:))','LineWidth',3,'Color',[1-ss/4 0 ss/4])
end
lgd=legend('Session 1','Session 2','Session 3');
lgd.Location='best';
lgd.FontSize=16;
xlabel('Trial Group')
xticklabels({'1','2-11','12-21','22-31'})
ylabel('Foresight')
%title('Build-Up')
xticks(1:5)
ylim([0 2.5])

%title('Update')
end


if(pp == 1 && strat == 1)
figure(1003)
subplot(1,3,tt)
for ss = 1:3 
plot(squeeze(ForeS(strat,ss,:))','LineWidth',3,'Color',[1-ss/4 0 ss/4])
hold on
end

switch type
    case 3 
    title('Build-Up')
    case 4
    title('Location Update')
    case 5
    title('Barrier Update')
    case 6
    title('Bar + GL')
end

set(gca,'FontSize',14)

if(type==3)

    text(1,0.5,'*','FontSize',28,'FontWeight','bold')
    text(1,1.1,'*','FontSize',28,'FontWeight','bold')
    text(2,1.1,'*','FontSize',28,'FontWeight','bold')
    text(3,1.1,'*','FontSize',28,'FontWeight','bold')
    
    lgd=legend('Session 1','Session 2','Session 3');
lgd.Location='best';
lgd.FontSize=16;
end

if(type==4)


    text(2,1.1,'*','FontSize',28,'FontWeight','bold')
    text(3,1.1,'*','FontSize',28,'FontWeight','bold')
    

end

if(type==5)


    text(1,0.5,'*','FontSize',28,'FontWeight','bold')
    
    text(2,1.1,'*','FontSize',28,'FontWeight','bold')
    text(3,1.1,'*','FontSize',28,'FontWeight','bold')
    

end

xlabel('Trial Group')
xticklabels({'1','2-11','12-21','22-31'})
ylabel('Foresight')
%title('Build-Up')
xticks(1:5)
ylim([0 2.5])
xlim([0.8 4.2])
if(type==3 )

    subplot(1,3,tt+1)
    plot(squeeze(ForeS(strat,1,:))','LineWidth',3,'LineStyle','--','Color',DC(1,:))
    hold on

    subplot(1,3,tt+2)
    plot(squeeze(ForeS(strat,1,:))','LineWidth',3,'LineStyle','--','Color',DC(1,:))
    hold on
    
end


end


end
end




end