% data_c1=C1_1();
% data_c2=C2_1();

data_c1=C1_2();
data_c2=C2_2();
%%
%�ֱ𽫱�Ǻõ�����������Ϊ����ѵ������
%data_c1 �� data_c2  
data1=[data_c1;data_c2];

%���½����ݴ�ɢ����ֲ�
sel1 = randperm(size(data1, 1));
data1=data1(sel1, :);
    
tseting_set=data1(1:1552,:); %data1ȡ30%��������Ϊ���Լ���
training_set=data1(1553:5175,:);%ȡ70��data1�е�������Ϊѵ����
    
    
    train_y=training_set(:,1:2);
    train_x=training_set(:,3:end);
    %���Լ����ݼ��ı�ǩ�����ݽ��з���
    test_y=tseting_set(:,1:2);
    test_x=tseting_set(:,3:end);
    
% save a1 train_x train_y test_x test_y
save a2 train_x train_y test_x test_y
%     X=data1(:,1);
%     y=data1(:,2:end);
% 
% save a X y

%������   5175��