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





Border_Map= ones(24,1)*2;
Border_Map([7 9 10 12 13 15])=1;
              
     
s=[];
t=[];
for node=1:size(Transition_Map,1)
    add_n=Transition_Map(node,:);
    add_n=add_n(add_n>0);
    for aa=1:numel(add_n)
        s=cat(2,s,node);
        t=cat(2,t,add_n(aa));
        
    end
    



end


G_Base=graph(s,t);           



G_Dir = digraph(s,t);
figure(2)
plot(G_Dir,'LineWidth',3,'NodeFontSize',20,'NodeFontWeight','bold','EdgeAlpha',0.6,'MarkerSize',10,'ArrowSize',15)

%%
Distance_Matrix=zeros(size(Transition_Map,1));

for ss=1:size(Transition_Map,1)
    for ee=1:size(Transition_Map,1)
    [~,Distance_Matrix(ss,ee)]=shortestpath(G_Base,ss,ee);
    
    
    
    end
    
    
end
%% 
A = readtable('animal_data_sheet_coh_1_2_3_4_5.xlsx','Sheet','raw');

%A = readtable('coh2_3_4_5_6_7 pharmaco.xlsx','Sheet','raw');

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

% Sh = A(:,11); %shortest path
% Sh = table2array(Sh);


F = A(:,6); %barrier
F=table2cell(F);
F = string(F);

Dr = A(:,30);
Dr = table2array(Dr); %if 1:drug if 0:vehicle

DistriComp = {};

 for typ=[3]
para_search=[];
     for session = 1:1
for tri=1:4

    if(tri==1)
    frst = C<=1 & C>0;
    else
        
frst = C<=1+10*(tri-1) & C>1+(tri-2)*10;% & Z==1; %1st trial 1st session
    end
rest = C~=1;% & Z==1; %all other trials sess1
allsess = Z==session; %all sess but s1
scnd = C==1 & Z==2; %S2T1

type = T==typ;
%type = T==3 | T==4;

St_List=S(find(type & allsess & frst),1);
En_List=E(find(type & allsess & frst),1);

DT_SP_root=D(find(type & allsess & frst),1); 

barriers = F(find(type & allsess & frst),1);

PP=P(find(type & allsess & frst),1); %paths per trials in group of 10s

Path=table2cell(PP); %Convert table to cells
%Path = rmmissing(Path);





%OD = DT_SP(DT_SP>1);
%DT_SP(isnan(DT_SP))=[];
% Data=D(find(type),1);   

% Shortest = Sh(find(type),1);
% Shortest(isnan(Shortest))=[];           

%% GENERATE RANDOM TRIALS             
 
str_ch=2;  %% CHOICE OF SEARCH-STRATEGY: 1=Random Foraging 2=Optimal Foraging


str_ch_pl = 1;


trial_len_limit=2; %% MINIMUM LENGTH OF THE TRIALS TO BE USED FOR ANALYSIS

%goal_detect_mean=0;

n_rep=10;    %for frst put 10 to have more random data 


 
g_tri = [0.5 0.2 0.2 0.2];  
p_tri = [0 0 0 0];
 
 %t_prob=0.6; %%0.105 Probability to switch from bordering to a diagonal run 0.3
for goal_detect_mean=g_tri(tri)
   
    if(str_ch_pl==1)
    prob_l = [0 0.1 0.15 0.2 0.25];
    elseif(str_ch_pl==2)
    prob_l = [0.1 0.15 0.2 0.25];   
    end
    
for prob= p_tri(tri)% prob_l
    t_prob=prob;

    
    DT_SP=DT_SP_root; 
    
    
n_trials=size(St_List,1);    
 %n_trials=1000;
 L_Trials=zeros(n_trials*n_rep,1);
 ShL_Trials=zeros(n_trials*n_rep,1);
 Model_Drift=zeros(n_trials*n_rep,1);
 
 
 Trial_Len=zeros(n_trials,1);
 Trial_Drift=zeros(n_trials,2);
 
  B_Beh = zeros(n_trials,1);
 B_Mod = zeros(n_trials,1);
 
 
 Trial_Choices=cell(n_trials,1);
 tr_r=0;

 Border_Beh = [];
    
 for tr=1:n_trials        %create random starting-ending locations     

     
     %START BARRIERS
     
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
     
G=G_barrier;     
     
Distance_Matrix=zeros(size(Transition_Map_Barrier,1));

for ss=1:size(Transition_Map_Barrier,1)
    for ee=1:size(Transition_Map_Barrier,1)
    [~,Distance_Matrix(ss,ee)]=shortestpath(G,ss,ee);
    
    
    
    end
    
    
end   
     
     
    % END BARRIERS 
     
     
     
     
     
  St_Node=St_List(tr);
  En_Node=En_List(tr);
 
[nodes_path,short]=shortestpath(G,St_Node,En_Node); %finds nodes path/shortest length

  %vector with all the shortest lengths
 Trial_Len(tr)=short;
  
 
 
    
    
    vec=str2num(Path{tr}); %convert it to numbers of followed nodes
    
    if(numel(vec)==0)
    continue
    
    end
    Border_Beh = cat(1,Border_Beh,Border_Map(vec));
    B_Beh(tr)=numel(find(Border_Map(vec)==1))/numel(Border_Map(vec));
    
    opt_path_cum=[];
    for rr=1:10
    ww=ones(size(s));
    ww=ww+rand(size(ww))*0.01; %randomly adds weight to some nodes so it will take all shortespaths randomly
        
        GG_Noise=digraph(s,t,ww);
    [opt_path,opt_d]=shortestpath(GG_Noise,vec(1),vec(end)); %finds the optimal path and distance for the st-end node of each path
    opt_path_cum=cat(1,opt_path_cum,opt_path(:));
    
    end
    opt_path=unique(opt_path_cum);
    
    
    dist_step=ones(numel(vec),1)*1000;
  
    for step=1:length(vec)
           
            for nn=1:numel(opt_path)
        [~,dist] = shortestpath(G,vec(step),opt_path(nn));
           dist_step(step) = min(dist,dist_step(step));
            end
           
        
           
    end
        
     Trial_Drift(tr,:) = [max(dist_step) mean(dist_step) ];    
        
 
 
 
 
 
 
 
 
 
 
 
 
 for rep=1:n_rep
 
     Border_Mod = [];
     
 tr_r=tr_r+1;
 ShL_Trials(tr_r)=short;
 Goal=0;
 Length=0;
 
 Model_Drift(tr_r) = 0;
 
 Visited=zeros(size(Transition_Map,1),1);
 
 In_Diag=0;
 
 Old_Dist=short;
 clear Choice
 while(Goal==0)  
     if(Length==0)  %if he hasn't moved yet,
     C_Node=St_Node;  %current node = starting node
     Last=St_Node;    %and last visited node is the starting node
     end
     Border_Mod=cat(1,Border_Mod,Border_Map(C_Node));  
     
     
     
        dist_model = 1000;
        for nn=1:numel(opt_path)
        [~,dist] = shortestpath(G,C_Node,opt_path(nn));
           dist_model = min(dist,dist_model);
        end
     
    Model_Drift(tr_r) = max(Model_Drift(tr_r),dist_model); 
     
     
     
     %Rnadom Foraging
     if(str_ch==1)
     
     ch=randi(3);
     Next=Transition_Map(C_Node,ch);  %random adjacent node
     
     while(Next==0 || Next==Last)   %if this random number is 0 or the last visited node
     ch=randi(3);                %find a new one next node
     Next=Transition_Map(C_Node,ch);
     end
     
     
     %Optimal Foraging
     elseif(str_ch==2)
     
     %Is the animal already on a digaonal run?     
     if(In_Diag==1)    
         
         
         Next=diag_nodes(dn+1);
         dn=dn+1;
         if(Next==Target)
         In_Diag=0;
         dn=0;
         end
     
     %Or is the animal on a random walk?    
     elseif(In_Diag==0)
         
         %Check if switching to a diagonal run 
         go_diag=(rand(1)<t_prob);
         
         if go_diag
         
         %Establish Target and trajectory     
         Pot_Targ=intersect(find(Border_Map==2),find(Distance_Matrix(C_Node,:)>3));  %>3  
         Target=Pot_Targ(randi(numel(Pot_Targ)));    
         diag_nodes=shortestpath(G,C_Node,Target);    
         avoid_attempt=0;
         %Try to avoid already visited nodes
         
             
%          while(sum(Visited(diag_nodes))>numel(diag_nodes) * 0.5)%*0.5)
%          
%          Target=Pot_Targ(randi(numel(Pot_Targ)));    
%          diag_nodes=shortestpath(G,C_Node,Target);
%          avoid_attempt=avoid_attempt+1;
%          
%          if(avoid_attempt>5) %5
%          go_diag=1;
%          break 
%         
%          end
%          
%          
%          end    
          
         if go_diag
         dn=1;
        Next=diag_nodes(dn+1);
         dn=dn+1;
         In_Diag=1;
         end
         end
         
         
         %Keep random walking
         if ~go_diag
                 
                if(str_ch_pl==1)
                 ch=randi(3);
                 Next=Transition_Map(C_Node,ch);  %random adjacent node
     
                 while(Next==0 || Next==Last)   %if this random number is 0 or the last visited node
                 ch=randi(3);                %find a new one next node
                 Next=Transition_Map(C_Node,ch);
                 end
                 %disp([Next rep tr])   
                
                
                elseif(str_ch_pl==2)
            Av_N=find(Transition_Map(C_Node,:)~=Last & Transition_Map(C_Node,:)~=0);
            B_Nodes=intersect(find(Border_Map==2),Transition_Map(C_Node,Av_N)); %==2
            Next=B_Nodes(1);
            %disp([Next rep tr])
            
            if(numel(B_Nodes)>1 && Border_Map(Last)==2 && Length>0)  %%%>1
                B_Nodes
            disp('Error in Routine')
            pause()
            end
            
                end
        end
     end
     
     
     end
     
     Visited(C_Node)=1;
     
     Last=C_Node;
     C_Node=Next;
     
     Length=Length+1;
     
     
     [~,New_Dist]=shortestpath(G,C_Node,En_Node);
     if(New_Dist<Old_Dist)
     
         Choice(Length)=1;
         
     else
        
         Choice(Length)=0;
     end
     
     goal_detect = exprnd(goal_detect_mean);
     
     if(Distance_Matrix(C_Node,En_Node)<=goal_detect) %<2 %C_Node==En_Node) %<1 only works with t_prob=0.1 and sum<numel*0.9
         Goal=1;
     end
     Old_Dist=New_Dist;
 end
     
 
 B_Mod(tr_r)=numel(find(Border_Mod==1))/numel(Border_Mod);
 
 L_Trials(tr_r)=Length+Distance_Matrix(C_Node,En_Node);
 Choice(Length+1:Length+Distance_Matrix(C_Node,En_Node))=1;
 
 Trial_Choices{tr_r}=Choice;
 end
 end

 
% Remove short trials  
 
Trial_Drift = Trial_Drift(:,1);


DT_SP(Trial_Len<trial_len_limit)=[];
Trial_Drift(Trial_Len<trial_len_limit)=[];
B_Beh(Trial_Len<trial_len_limit)=[];




L_Trials(ShL_Trials<trial_len_limit)=[];
ShL_Trials(ShL_Trials<trial_len_limit)=[];
Model_Drift(ShL_Trials<trial_len_limit)=[]; 
 B_Mod(ShL_Trials<trial_len_limit)=[]; 
 
 
 L_Ratio=L_Trials./ShL_Trials;
 L_Ratio(find(L_Ratio==Inf))=[];

    OD = DT_SP(DT_SP>1.1);
    LR = L_Ratio(L_Ratio>1.1);

GoDir_Dat=(numel(DT_SP)-numel(OD))/numel(DT_SP);
GoDir_Sim=(numel(L_Ratio)-numel(LR))/numel(L_Ratio);

 [h,p,k]=kstest2(L_Ratio,DT_SP);
 
 P_Vec=linspace(1,10,16);
 
 

 
 P1=histcounts(L_Ratio,'Binedges',P_Vec,'Normalization','probability')+0.001;
 P2=histcounts(DT_SP,'Binedges',P_Vec,'Normalization','probability')+0.001;
 
 P1=P1/sum(P1);
 P2=P2/sum(P2);
 
 KL=kldiv(P_Vec(2:end),P2,P1);
 
 
  [h_dr,p_dr,k_dr]=kstest2(Trial_Drift,Model_Drift);
 
 P_Vec=linspace(min(cat(1,Trial_Drift,Model_Drift)),max(cat(1,Trial_Drift,Model_Drift)),16);
 
 

 
 P1=histcounts(Model_Drift,'Binedges',P_Vec,'Normalization','probability')+0.001;
 P2=histcounts(Trial_Drift,'Binedges',P_Vec,'Normalization','probability')+0.001;
 
 P1=P1/sum(P1);
 P2=P2/sum(P2);
 
 KL_dr=kldiv(P_Vec(2:end),P2,P1);
 
 
 B1=numel(find(Border_Beh==1))/numel(Border_Beh);
 B2=numel(find(Border_Mod==1))/numel(Border_Mod);
 

 DistriComp{1,tri} = L_Ratio;
 DistriComp{2,tri} = DT_SP;
 DistriComp{3,tri} = Model_Drift;
 DistriComp{4,tri} = Trial_Drift;
 DistriComp{5,tri} = B_Mod;
 DistriComp{6,tri} = B_Beh;   
 

figure(tri)
subplot(3,1,1)
histogram(L_Ratio,'Binedges',[0:0.5:20],'FaceAlpha',0.3,'FaceColor',[1 0 0],'Normalization','cdf')
hold on
histogram(DT_SP,'Binedges',[0:0.5:20],'FaceAlpha',0.3,'FaceColor',[0 0 1],'Normalization','cdf')
legend('Log10Random','Log10Data')

subplot(3,1,2)
histogram(Model_Drift,'Binedges',[0:1:8],'FaceAlpha',0.3,'FaceColor',[1 0 0],'Normalization','cdf')
hold on
histogram(Trial_Drift,'Binedges',[0:1:8],'FaceAlpha',0.3,'FaceColor',[0 0 1],'Normalization','cdf')
legend('Log10Random','Log10Data')

subplot(3,1,3)
histogram(B_Mod,'Binedges',[0:0.1:1],'FaceAlpha',0.3,'FaceColor',[1 0 0],'Normalization','cdf')
hold on
histogram(B_Beh,'Binedges',[0:0.1:1],'FaceAlpha',0.3,'FaceColor',[0 0 1],'Normalization','cdf')
legend('Log10Random','Log10Data')


end
end



end
     end

 end

 
 save('DistributionComparison','DistriComp')
 