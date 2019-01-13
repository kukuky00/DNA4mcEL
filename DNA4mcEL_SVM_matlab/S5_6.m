function [M]=S5_6()
M=zeros(1812,30*40);
A=textread('S5.txt','%s');
B=char(A);
for i=1:1812
    E(i,:)=B(i*2,:);
end

b=size(E,1);      %核小体的样本的个数（即列数）
a=41;
for i=1:b
    for j=1:a-1
        %一级结构
        m=0;
        if E(i,j)=='A'
           M(i,(j*30))=1;           
        elseif E(i,j)=='T'
           M(i,(j*30-1))=1;
        elseif E(i,j)=='C'
           M(i,(j*30-2))=1;
        elseif E(i,j)=='G'
           M(i,(j*30-3))=1;                                  
        end
        for l=1:size(E,2)-1
            if E(i,l)==E(i,j)
                m=m+1;
            end
        end
        M(i,(j*30-4))=m/41;
        %二级结构
        
        m=0;n=0;s=0;b=0;c=0;d=0;e=0;f=0;g=0;
        if E(i,j)=='A'
            if E(i,j+1)=='A'
                k=5;       b =0.06; c=0.50; d=0.09; e=1.59; f=0.11;g=-0.11;
            elseif E(i,j+1)=='T'
                k=6;       b =1.07; c=0.22; d=0.83; e=-1.02; f=2.51;g=1.17;
            elseif E(i,j+1)=='G'
                k=7;       b =0.78; c=0.36; d=-0.28; e=0.68; f=-0.24;g=-0.62;
            elseif E(i,j+1)=='C'
                k=8;       b =1.50; c=0.50; d=1.19; e=0.13; f=1.29;g=1.04;
            end
        elseif E(i,j)=='T'
            if E(i,j+1)=='A'
                k=9;        b =-1.23; c=-2.37; d=-1.38; e=-2.24; f=-1.51;g=-1.39;
            elseif E(i,j+1)=='T'
                k=10;       b =0.06; c=0.50; d=0.09; e=1.59; f=0.11;g=-0.11;
            elseif E(i,j+1)=='G'
                k=11;       b =-1.38; c=-1.36; d=-1.01; e=-0.86; f=-0.62;g=-1.25;
            elseif E(i,j+1)=='C'
                k=12;       b = -0.08; c=0.50; d=0.09; e=0.13; f=-0.39;g=0.71;
            end
        elseif E(i,j)=='G'
            if E(i,j+1)=='A'
                k=13;        b = -0.08; c=0.50; d=0.09; e=0.13; f=-0.39;g=0.71;
            elseif E(i,j+1)=='T'
                k=14;        b = 1.50; c=0.50; d=1.19; e=0.13; f=1.29;g=1.04;
            elseif E(i,j+1)=='G'
                k=15;        b =  0.06; c=1.08; d=-0.28; e=0.56; f=-0.82;g=0.24;
            elseif E(i,j+1)=='C'
                k=16;        b = -0.08; c=0.22; d=2.30; e=-0.35; f=0.65;g=1.59;
            end
        elseif E(i,j)=='C'
            if E(i,j+1)=='A'
                k=17;         b =  -1.38; c=-1.36; d=-1.01; e=-0.86; f=-0.62;g=-1.25;
            elseif E(i,j+1)=='T'
                k=18;         b = 0.78; c=0.36; d=-0.28; e=0.68; f=-0.24;g=-0.62;
            elseif E(i,j+1)=='G'
                k=19;         b =  -1.66; c=-1.22; d=-1.38; e=-0.82; f=-0.29;g=-1.39;
            elseif E(i,j+1)=='C'
                k=20;         b =0.06; c=1.08; d=-0.28; e=0.56; f=-0.82;g=0.24;
            end
        end
        M(i,j*30-k)=1;
        M(i,j*30-21)=g;
        M(i,j*30-22)=f;
        M(i,j*30-23)=e;
        M(i,j*30-24)=d;
        M(i,j*30-25)=c;
        M(i,j*30-26)=b;
%         M(i,j*31-27)=a;
        
        
        if E(i,j)=='G'||E(i,j)=='A'
            M(i,j*30-27)=1;
        end
        
        if E(i,j)=='T'||E(i,j)=='A'
            M(i,j*30-28)=1;
        end
        
        if E(i,j)=='A'||E(i,j)=='C'
            M(i,j*30-29)=1;
        end
                
    end
end
%     M(1:495,size(M,2)-1)=1; M(496:size(M,1),size(M,2))=1;

%%%%%%%%%%%%%%%%%%%%%%%
%  M(1:(size(M,1)/2),size(M,2)-1)=1; M(((size(M,1)/2)+1):size(M,1),size(M,2))=1;
%%%%%%%%%%%%%%%%%%%%%%%%



%  M(1:133,size(M,2))=1;
%将ATCG的每一位编译成新的四位数值
a=ones(906,1);
b=zeros(906,1);
c=[a;b];
M=[c,M];





