%% Species:   G.subterraneus   ------Acc=81.53%
% Benchmark dataset S5 contains 906 4mC site containing sequences and
% 906 non-4mC site containing sequences of G. subterraneus.
close all;
clear;
clc;

%第一次写的编译文本
% data_s=S5_1();

%第二次写的编译文本
% data_s=S5_2();

%第三次写的编译文本
% data_s=S5_4();

%第四次写的编译文本
% data_s=S5_6();

%第五次写的编译文本
data_s=S5_7();
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
    model=svmtrain(training_label,training_data,'-t 3' );
%     model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
 %%第一次写的编译文本
% data_s=S5_1();
    %-s     % 当值为0   accuracy =80%     % 当值为1   accuracy =80%
    %-t     % 当值为0   accuracy =77%     % 当值为1   accuracy =71.1921%
            % 当值为2   accuracy =81%     % 当值为3   accuracy =80.298%
%%  ***********************************************************************************   
%%第二次写的编译文本
% data_s=S5_2();
    %-s     % 当值为0   accuracy =79%          % 当值为1   accuracy =80%
    %-t     % 当值为0   accuracy =74.5033%     % 当值为1   accuracy =73.6755%
            % 当值为2   accuracy =79.1391%     % 当值为3   accuracy =78.8079% 
%%  ***********************************************************************************   
%%第三次写的编译文本
% data_s=S5_4();
    %-s     % 当值为0   accuracy =80.7947%      % 当值为1   accuracy =81.298%
    %-t     % 当值为0   accuracy =74.8344%      % 当值为1   accuracy =48.0132%
            % 当值为2   accuracy =81.298%       % 当值为3   accuracy =80.4636%
%%  ***********************************************************************************   
%%第四次写的编译文本
% data_s=S5_6();
    %-s     % 当值为0   accuracy =81.457%       % 当值为1   accuracy =82%
    %-t     % 当值为0   accuracy =73.5099%      % 当值为1   accuracy =49.8344%
            % 当值为2   accuracy =80.106%       % 当值为3   accuracy =81.4437%
%%  ***********************************************************************************   
    %  %第五次写的编译文本
% data_s=S5_7();
    %-s     % 当值为0   accuracy =79.1457%     % 当值为1   accuracy =81.8325%
    %-t     % 当值为0   accuracy =75.8278%     % 当值为1   accuracy =74.6689%
            % 当值为2   accuracy =80%          % 当值为3   accuracy =78.1457%
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
