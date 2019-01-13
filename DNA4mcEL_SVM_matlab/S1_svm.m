%% Species:   C.elegans ------Acc=78.04 %
% Benchmark dataset S1 contains 1554 4mC site containing sequences and 1554
% non-4mC site containing sequences of C. elegans.
close all;
clear;
clc;

%第一次写的编译文本
data_s=S1_1();

%第二次写的编译文本
% data_s=S1_2();

%第三次写的编译文本
% data_s=S1_3();

%第四次写的编译文本
% data_s=S1_4();

%第五次写的编译文本
% data_s=S1_5();

%第六次写的编译文本
% data_s=S1_6();

%第七次写的编译文本
% data_s=S1_7();

%第8次写的编译文本
% data_s=S1_8();

%第9次写的编译文本
% data_s=S1_9();
%%
%使用交叉验证函数的方法
% Data = rand(9,3);%创建维度为9×3的随机矩阵样本
% indices = crossvalind('Kfold', 9, 3);%将数据样本随机分割为3部分
% for i = 1:3 %循环3次，分别取出第i部分作为测试样本，其余两部分作为训练样本
%     test = (indices == i);
%     train = ~test;
%     trainData = Data(train, :);
%     testData = Data(test, :);
% end
[n1,m1]=size(data_s);

indices = crossvalind('Kfold',n1, 3);%将数据样本随机分割为3部分

for i = 1:3
    %循环3次，分别取出第i部分作为测试样本，其余两部分作为训练样本
    test = (indices == i);
    train = ~test;
    trainData = data_s(train, :);
    testData = data_s(test, :);
    
   %%
%    归一化函数:scaleForSVM                               [train_scale,test_scale,ps]= scaleForSVM(train_data,test_data,ymin,ymax)
% 输入：
% train_data:训练集，格式要求与svmtrain相同。               test_data:测试集，格式要求与svmtrain相同。
% ymin，ymax:归一化的范围，即将训练集和测试都归一化到[ymin,ymax]，这两个参数可不输入，默认值为ymin=0，ymax=1，即默认将训练集和测试都归一化到[0,1]。
% 输出：
% train_scale:归一化后的训练集。                            test_scale:归一化后的测试集。
   ymin=0;
   ymax=1;
   [train_scale,test_scale,ps]= scaleForSVM(trainData,testData,ymin,ymax);



%% 降维预处理(pca)
%  [train_scale,test_scale] = pcaForSVM(train_scale,test_scale,97);
%  处理的效果不好
   %%
    %重新将数据打散随机分布
    % sel1 = randperm(size(data1, 1));
    % data1=data1(sel1, :);
    % sel2 = randperm(size(data2, 1));
    % data2=data2(sel2, :);
    % sel3 = randperm(size(data3, 1));
    % data3=data3(sel3, :);

   %%
    %将数据重新划分为训练集和测试集
    %对第一个C. elegan生物DNA数据进行分析
    % tseting_set=data1(1:1552,:); %data1取30%的数据作为测试集合
    % training_set=data1(1553:3623,:);%取70的data1中的数据作为训练集
    % 
    %对训练集数据集的标签和数据进行分离
    training_label=train_scale(:,1);
    training_data=train_scale(:,2:end);
    %测试集数据集的标签和数据进行分离
    tseting_lable=test_scale(:,1);
    tseting_data=test_scale(:,2:end);


   %%
    model=svmtrain(training_label,training_data,'-s 1' );
%     model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
     
%%第一次写的编译文本
% data_s=S1_1();
    %-s
    % 当值为0   accuracy_L =78.2819%      % 当值为1   accuracy =80.4054% 
    % 当值为2    accuracy =22.2973%       % 当值为3  and  % 当值为4  is wrong 
    %-t 核函数类型
    % 当值为0   accuracy_L = 81.3707%    % 当值为1   accuracy =72.3938% 
    % 当值为2    accuracy =77.8958%      % 当值为3    accuracy =76.5444%
    %-d 核函数类型      accuracy = 77.4485
%%  ***********************************************************************************   
%%第二次写的编译文本
% data_s=S1_2();
    %-s
    % 当值为0   accuracy_L =76.6409%       % 当值为1   accuracy = 81.0811% 
    % 当值为2    accuracy =23.8417%      % 当值为3  and  % 当值为4  is wrong 
    %-t 核函数类型
    % 当值为0   accuracy_L =  78.7645%    % 当值为1   accuracy =46.7181% 
    % 当值为2   accuracy =76.9305%        当值为3    accuracy =76.6409%
    
%%  ***********************************************************************************   
%%第三次写的编译文本
% data_s=S1_3();
    %-s
    % 当值为0   accuracy_L =78.0888%       % 当值为1   accuracy = 82.1776% 
    % 当值为2    accuracy = 23.3591%      % 当值为3  and  % 当值为4  is wrong 
    %-t 核函数类型
    % 当值为0   accuracy_L = 79.0541%      % 当值为1   accuracy = 47.7799%
    % 当值为2    accuracy = 78.3784%      % 当值为3    accuracy =75.5444%

%%  ***********************************************************************************   
%%第四次写的编译文本
% data_s=S1_4();
    %-s
    % 当值为0   accuracy_L = 78.861%    % 当值为1   accuracy =82.3359%
    % 当值为2    28.2819%     % 当值为3  and  % 当值为4  is wrong 
    %-t 核函数类型
    % 当值为0   accuracy_L =79.2471%   % 当值为1   accuracy = 49.5174% 
    % 当值为2    accuracy =  79.7297%      % 当值为3    accuracy =78.0888% 
%%第五次写的编译文本
% data_s=S1_5();
    %-s
    % 当值为0   accuracy_L = 79.0541%    % 当值为1   accuracy =82.3359% 
    % 当值为2    28.2819%     % 当值为3  and  % 当值为4  is wrong 
    %-t 核函数类型
    % 当值为0   accuracy_L =77.5097%   % 当值为1   accuracy =  49.0347%
    % 当值为2    accuracy =  79.8263%      % 当值为3    accuracy =78.0888%     
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
