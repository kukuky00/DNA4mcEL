% data_c1=C1_1();
% data_c2=C2_1();

data_c1=C1_2();
data_c2=C2_2();
%%
%分别将标记好的三组数据作为三个训练集合
%data_c1 和 data_c2  
data1=[data_c1;data_c2];

%重新将数据打散随机分布
sel1 = randperm(size(data1, 1));
data1=data1(sel1, :);
    
tseting_set=data1(1:1552,:); %data1取30%的数据作为测试集合
training_set=data1(1553:5175,:);%取70的data1中的数据作为训练集
    
    
    train_y=training_set(:,1:2);
    train_x=training_set(:,3:end);
    %测试集数据集的标签和数据进行分离
    test_y=tseting_set(:,1:2);
    test_x=tseting_set(:,3:end);
    
% save a1 train_x train_y test_x test_y
save a2 train_x train_y test_x test_y
%     X=data1(:,1);
%     y=data1(:,2:end);
% 
% save a X y

%共计有   5175列