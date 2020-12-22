%% PREPARE GRAPH
clear all
close all
Transition_Map = [6 2 0;
                  1 3 0   
                  2 7 4
                  3 0 5
                  4 0 8
                  1 9 17
                  3 9 10
                  5 10 11
                  6 7 12
                  7 8 13
                  8 14 0
                  19 15 9
                  10 15 16
                  11 16 0
                  12 22 13
                  13 24 14
                  6 18 0
                  17 19 0
                  18 12 20
                  0 19 21
                  0 20 22
                  15 21 23
                  0 22 24
                  16 23 0];








%% 
A = readtable('animal_data_sheet_coh_1_2_3_4_5.xlsx','Sheet','raw');
S = A(:,8); %Start location
S=table2array(S);
D = A(:,14); %DTSP
D=table2array(D);


T=A(:,24); %type
T=table2array(T);
E=A(:,7); %End location
E=table2array(E);

C = A(:,3); %Trial
C=table2array(C);
Z = A(:,26); %Session
Z=table2array(Z);
R = A(:,25);
R = table2array(R); %Repeat

P=A(:,10); %path
N=A(:,16); %choices

F = A(:,6); %barrier
F=table2cell(F);
F = string(F);

%% 

All_means_Tr_one=[];
f=100;

 for type=3:6 %[3 4]
para_search=[];
 All_means=[];
Direct_P = zeros(3,4);
Trial_Length = {};
Fits_Para=[];

     for sess = 1:3
for tri=1:4

%     if(tri==1)
%     frst = C<=1 & C>0;
%     else
        
 if(tri==1)
    frst = C<=1 & C>0;
    else
        
frst = C<=1+10*(tri-1) & C>1+(tri-2)*10;% & Z==1; %1st trial 1st session
    end


%frst = C<(tri-1)*10+10 & C>(tri-1)*10;
% frst = C<=1+10*(tri-1) & C>1+(tri-2)*10;% & Z==1; %1st trial 1st session
%     end

allsess = Z==sess; %all sess but s1


St_List=S(find(T==type & allsess & frst),1);
En_List=E(find(T==type & allsess & frst),1);

 barriers = F(find(T==type & allsess & frst),1);        

DT_SP_root=D(find(T==type & allsess & frst),1);  
 
% COMPUTE THE DISTANCE NORMALIZATON FOR THIS SET OF TRIALS            

n_trials=size(St_List,1);    

D_Trial_Mea=zeros(n_trials,1);

 for tr=1:n_trials        %create random starting-ending locations     

  St_Node=St_List(tr);
  En_Node=En_List(tr);
 
  
  PP=P(find(T==type & allsess & frst),1); %paths per trials in group of 10s

NN=N(find(T==type & allsess & frst),1); %choices


Path=table2cell(PP); %Convert table to cells
Choices = table2cell(NN);

Path = rmmissing(Path);
% Choices = rmmissing(Choices);

dist_m_all={};
dist_m_good={}; %correct 1st choice(=1)
dist_m_bad={};  %wrong 1st choice(=0)
j=0;
h=0;
k=[];

    if barriers(tr,1)=="bar1"
    Barrier_Position = [17 18; 11 14; 15 22];
    elseif barriers(tr,1)=="bar2"
    Barrier_Position = [4 5; 7 9; 20 21];  
    elseif barriers(tr,1)=="bar3" 
    Barrier_Position = [1 2; 13 15; 23 24];
    elseif barriers(tr,1)=="bar4" 
    Barrier_Position = [7 10; 8 11; 18 19];
    elseif barriers(tr,1)=="bar5" 
    Barrier_Position = [3 4; 12 15; 16 24];
    elseif barriers(tr,1)=="bar6" 
    Barrier_Position = [5 8; 9 12; 21 22];
    elseif barriers(tr,1)=="bar7" 
    Barrier_Position = [6 9; 13 16; 19 20];
    elseif barriers(tr,1)=="bar8" 
    Barrier_Position = [2 3; 10 13; 22 23];
    elseif barriers(tr,1)=="bar9" 
    Barrier_Position = [3 7; 6 17; 14 16];
    elseif barriers(tr,1)=="bar10" 
    Barrier_Position = [1 6; 8 10; 15 22];
    elseif barriers(tr,1)=="bar11" 
    Barrier_Position = [7 9; 12 19; 23 24];
    elseif barriers(tr,1)=="bar12" 
    Barrier_Position = [1 2; 12 15; 11 14];
    elseif barriers(tr,1)=="bar13" 
    Barrier_Position = [17 19; 13 15; 8 11];
    elseif barriers(tr,1)=="bar14" 
    Barrier_Position = [1 2; 9 12; 11 14];
else
     Barrier_Position = [];   
end 
    
    
Transition_Map_Barrier = Transition_Map;
for bb=1:size(Barrier_Position,1)
    
  To_Eliminate=find(Transition_Map_Barrier(Barrier_Position(bb,1),:)==Barrier_Position(bb,2));
  Transition_Map_Barrier(Barrier_Position(bb,1),To_Eliminate)=0;
  To_Eliminate=find(Transition_Map_Barrier(Barrier_Position(bb,2),:)==Barrier_Position(bb,1));
  Transition_Map_Barrier(Barrier_Position(bb,2),To_Eliminate)=0;
end
 
s=[];
t=[];
for node=1:size(Transition_Map_Barrier,1)
  add_n=Transition_Map_Barrier(node,:);
  add_n=add_n(add_n>0);
  for aa=1:numel(add_n)
    s=cat(2,s,node);
    t=cat(2,t,add_n(aa));
  end
end
G_barrier=graph(s,t);

Distance_Matrix=zeros(size(Transition_Map_Barrier,1));

for ss=1:size(Transition_Map_Barrier,1)
    for ee=1:size(Transition_Map_Barrier,1)
    [~,Distance_Matrix(ss,ee)]=shortestpath(G_barrier,ss,ee);
    
    
    
    end
    
    
end


    opt_path_cum=[];
    for rr=1:20
    ww=ones(size(s));
    ww=ww+rand(size(ww))*0.01; %randomly adds weight to some nodes so it will take all shortespaths randomly
        
    GG_Noise=digraph(s,t,ww);
    [opt_path,opt_d]=shortestpath(GG_Noise,St_Node,En_Node); %finds the optimal path and distance for the st-end node of each path
    opt_path_cum=cat(1,opt_path_cum,opt_path(:));
    
    end
    opt_path=unique(opt_path_cum);




%[nodes_path,short]=shortestpath(G_barrier,St_Node,En_Node); %finds nodes path/shortest length

%Let's take the distance matrix of all the points to the optimal path
D_Path=Distance_Matrix(opt_path,:);


%Mean distance of nodes from the optimal path (This might be changed into the maximal distance, depending on what works best)
D_Trial_Mea(tr)=mean(min(D_Path,[],1));

 end

 %This is the score you need to normalize the results 
 D_Normalization=mean(D_Trial_Mea);
 
 
%  
%  figure(11)
%  subplot(3,1,tri)
%  histogram(D_Trial_Mea,100)
 
 
 
 


 
 
 
 
 for i=1:length(Path) %iterate through the paths per trials
    
     
     
    vec=str2num(Path{i}); %convert it to numbers of followed nodes
    opt_path_cum=[];
    for rr=1:20
    ww=ones(size(s));
    ww=ww+rand(size(ww))*0.01; %randomly adds weight to some nodes so it will take all shortespaths randomly
        
        GG_Noise=digraph(s,t,ww);
    [opt_path,opt_d]=shortestpath(GG_Noise,vec(1),vec(end)); %finds the optimal path and distance for the st-end node of each path
    opt_path_cum=cat(1,opt_path_cum,opt_path(:));
    
    end
    opt_path=unique(opt_path_cum);
    
    
    dist_m=ones(numel(vec),1)*100;
    ts=0;
    for step=1:length(vec)
        
       if (~ismember(500,Transition_Map_Barrier(vec(step),:)) || step==1) %if 0:takes only the 3way nodes (for all nodes put a diff number like 50) AND the 1st choice no matter if it is 2 or 3
          ts=ts+1;
        for nn=1:numel(opt_path)
        [~,dist] = shortestpath(G_barrier,vec(step),opt_path(nn));
        if(dist<dist_m(step))
        dist_m(ts)=dist;
         end
        end
       end
    end
    
    dist_m(dist_m==100)=[];
    
    dec_vec=1:numel(dist_m);  %scale all paths to have the same length
    dec_vec=dec_vec/numel(dist_m)*6+1; %here i.e in 7 bins (the plus 1 is so that it always starts with 1)
    dec_vec=round(dec_vec); %round it so they all fall into the 7 bins
    
    
    dist_m_norm=zeros(7,1);
    for bb=1:7
        t_poi=find(dec_vec==bb);
        if(numel(t_poi)>0)
     dist_m_norm(bb)=mean(dist_m(t_poi));  %i.e if we wanted to fit 14 nodes to 7, each one of the 7 would take the mean of two
        end
        
        
    end
    
    dist_m_all{i} = dist_m;
    
%     dist_m_all{i}=dist_m_norm/D_Trial_Mea(i); %change to dist_m_norm to do this for the scaled path or to dist_m for the actual path
%     if(numel(str2num(Choices{i}))>0)
%     if(str2num(Choices{i}(1))==1)
%         j=j+1;
%     dist_m_good{j}=dist_m_norm; %and this
%     elseif(str2num(Choices{i}(1))==0)
%             h=h+1;
%     dist_m_bad{h}=dist_m_norm; %and this
%     
%     
%     end
%     end
    
    
    
 end


 Grand_mean=zeros(50,1);
  Grand_mean_c=zeros(50,1);
 kk=0;
 
 dist_m_use=dist_m_all; %choose correct(good) or wrong(bad) 1st choice
 
 for tt=1:numel(dist_m_use)
    if(numel(dist_m_use{tt})<50)
    kk=kk+1;
        Grand_mean(1:numel(dist_m_use{tt}))=Grand_mean(1:numel(dist_m_use{tt}))+dist_m_use{tt};%./D_Normalization;
        Grand_mean_c(:)=Grand_mean_c(:)+1;%./D_Normalization;
    
    end
     
     
 
 
 end
 
 %Grand_mean=Grand_mean./kk;

 Grand_mean = Grand_mean./Grand_mean_c;
 
 % Grand_mean=Grand_mean./D_Normalization;
All_means = [All_means , Grand_mean];

 
figure(type)
subplot(2,2,sess)
hold on
 plot(Grand_mean,'LineWidth',2,'Color',[1-tri/4 0 tri/4])
xlabel('Node choices')
ylabel('Mean')
ylim([0 3])
xlim([0 50]) %only for the scaled paths
%title(['Distance from ideal path (Wrong First Turn) Session 1-4 (All nodes) ' num2str(sess)])
switch type 
    case 3
title(['Session ', num2str(sess), ' Build-up '])
    case 4
title(['Session ', num2str(sess), ' Update '])
end
%grid minor

%All_means = Grand_mean; 
x_label = (1:size(All_means,1))';
% k=0;



y = Grand_mean;
   
F_Fit = @(x,xdata)x(1)*1/(exp(-2*log(x(3)/x(2))/(1-(x(2)/x(3)).^2))-exp(-2*log(x(3)/x(2))/((x(3)/x(2)).^2-1)))*(exp(-xdata.^2/(2*x(2).^2))-exp(-xdata.^2/(2*x(3).^2)));

x0 = [1.5 5 1];
lb = [0 0 0];
ub = [Inf Inf Inf];
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
[x] = lsqcurvefit(F_Fit,x0,x_label,y,lb,ub,options);
Fits_Para=cat(1,Fits_Para,x);
figure(100)
subplot(2,2,sess)
plot(y,'r')
hold on
plot(F_Fit(x,x_label),'b')
title(['Session', num2str(sess)])
hold on
%pause()







if(tri==1)
All_means_Tr_one = [All_means_Tr_one , Grand_mean];
figure(type)
subplot(2,2,4)
hold on
plot(Grand_mean,'LineWidth',2,'Color',[1-sess/4 0 sess/4])
xlabel('Nodes choices')
ylabel('Mean')
ylim([0 3])
xlim([0 50]) %only for the scaled paths
title(['First 10 Trials of Each Session'])
grid minor
end


figure(500+type)
HH = histcounts(DT_SP_root,1:0.5:5,'Normalization','probability');

subplot(2,2,sess)
plot(1:0.5:4.5,HH,'LineWidth',2,'Color',[1-tri/4 0 tri/4])
xlabel('Relative Trial Length')
ylabel('Probability')
title(['DTSP Session ' num2str(sess)])
hold on


Direct_P(sess,tri) = HH(1);
Trial_Length{sess,tri} = DT_SP_root;


end
     end

     
     save(['DistFromOpt_' num2str(type)],'All_means','Fits_Para','Trial_Length','Direct_P')
     
     

  figure(500+type)
  subplot(2,2,4)
  for sess=1:3
plot(Direct_P(sess,:),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
hold on
  end
xlabel('Trial Group')
ylabel('P of Direct Run')
title('Probability DTSP<1.5')
legend('Session 1','Session 2','Session 3')
hold on   
     

 figure(type )
 subplot(2,2,1)
 legend('Trial 1','Trials 2-11','Trials 12-21','Trials 22-31')
 
 figure(type)
  subplot(2,2,4)
 legend('First Trial, Session 1','First Trial, Session 2','First Trial, Session 3')
 
 
 figure(600+type)
 clf;
 for pp=1:3
    subplot(1,3,pp)
    FP = reshape(Fits_Para(:,pp),[],3);
    for sess = 1:3
     plot(smooth(FP(:,sess)),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
     hold on
    end
     xlabel('Trial Group')
ylabel('Fit Value')
switch pp
    case 1
        title('Peak')
    case 2
        title('Descending Length')
        
    case 3
         title('Ascending Length')
end
legend('Session 1','Session 2','Session 3')
     
 end
 
 
 
 
end
%     

 