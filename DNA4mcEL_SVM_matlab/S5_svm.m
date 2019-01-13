%% Species:   G.subterraneus   ------Acc=81.53%
% Benchmark dataset S5 contains 906 4mC site containing sequences and
% 906 non-4mC site containing sequences of G. subterraneus.
close all;
clear;
clc;

%��һ��д�ı����ı�
% data_s=S5_1();

%�ڶ���д�ı����ı�
% data_s=S5_2();

%������д�ı����ı�
% data_s=S5_4();

%���Ĵ�д�ı����ı�
% data_s=S5_6();

%�����д�ı����ı�
data_s=S5_7();
%%
%ʹ�ý�����֤�����ķ���
% Data = rand(9,3);%����ά��Ϊ9��3�������������
% indices = crossvalind('Kfold', 9, 3);%��������������ָ�Ϊ3����
% for i = 1:3 %ѭ��3�Σ��ֱ�ȡ����i������Ϊ����������������������Ϊѵ������
%     test = (indices == i);
%     train = ~test;
%     trainData = Data(train, :);
%     testData = Data(test, :);
% end
[n1,m1]=size(data_s);

indices = crossvalind('Kfold',n1, 3);%��������������ָ�Ϊ3����
for i = 1:3
    %ѭ��3�Σ��ֱ�ȡ����i������Ϊ����������������������Ϊѵ������
    test = (indices == i);
    train = ~test;
    trainData = data_s(train, :);
    testData = data_s(test, :);
    
   %%
%    ��һ������:scaleForSVM                               [train_scale,test_scale,ps]= scaleForSVM(train_data,test_data,ymin,ymax)
% ���룺
% train_data:ѵ��������ʽҪ����svmtrain��ͬ��               test_data:���Լ�����ʽҪ����svmtrain��ͬ��
% ymin��ymax:��һ���ķ�Χ������ѵ�����Ͳ��Զ���һ����[ymin,ymax]�������������ɲ����룬Ĭ��ֵΪymin=0��ymax=1����Ĭ�Ͻ�ѵ�����Ͳ��Զ���һ����[0,1]��
% �����
% train_scale:��һ�����ѵ������                            test_scale:��һ����Ĳ��Լ���
   ymin=0;
   ymax=1;
   [train_scale,test_scale,ps]= scaleForSVM(trainData,testData,ymin,ymax);

%% ��άԤ����(pca)
%  [train_scale,test_scale] = pcaForSVM(train_scale,test_scale,97);
%  �����Ч������
   %%
    %���½����ݴ�ɢ����ֲ�
    % sel1 = randperm(size(data1, 1));
    % data1=data1(sel1, :);
    % sel2 = randperm(size(data2, 1));
    % data2=data2(sel2, :);
    % sel3 = randperm(size(data3, 1));
    % data3=data3(sel3, :);

   %%
    %���������»���Ϊѵ�����Ͳ��Լ�
    %�Ե�һ��C. elegan����DNA���ݽ��з���
    % tseting_set=data1(1:1552,:); %data1ȡ30%��������Ϊ���Լ���
    % training_set=data1(1553:3623,:);%ȡ70��data1�е�������Ϊѵ����
    % 
    %��ѵ�������ݼ��ı�ǩ�����ݽ��з���
    training_label=train_scale(:,1);
    training_data=train_scale(:,2:end);
    %���Լ����ݼ��ı�ǩ�����ݽ��з���
    tseting_lable=test_scale(:,1);
    tseting_data=test_scale(:,2:end);


   %%
    model=svmtrain(training_label,training_data,'-t 3' );
%     model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
 %%��һ��д�ı����ı�
% data_s=S5_1();
    %-s     % ��ֵΪ0   accuracy =80%     % ��ֵΪ1   accuracy =80%
    %-t     % ��ֵΪ0   accuracy =77%     % ��ֵΪ1   accuracy =71.1921%
            % ��ֵΪ2   accuracy =81%     % ��ֵΪ3   accuracy =80.298%
%%  ***********************************************************************************   
%%�ڶ���д�ı����ı�
% data_s=S5_2();
    %-s     % ��ֵΪ0   accuracy =79%          % ��ֵΪ1   accuracy =80%
    %-t     % ��ֵΪ0   accuracy =74.5033%     % ��ֵΪ1   accuracy =73.6755%
            % ��ֵΪ2   accuracy =79.1391%     % ��ֵΪ3   accuracy =78.8079% 
%%  ***********************************************************************************   
%%������д�ı����ı�
% data_s=S5_4();
    %-s     % ��ֵΪ0   accuracy =80.7947%      % ��ֵΪ1   accuracy =81.298%
    %-t     % ��ֵΪ0   accuracy =74.8344%      % ��ֵΪ1   accuracy =48.0132%
            % ��ֵΪ2   accuracy =81.298%       % ��ֵΪ3   accuracy =80.4636%
%%  ***********************************************************************************   
%%���Ĵ�д�ı����ı�
% data_s=S5_6();
    %-s     % ��ֵΪ0   accuracy =81.457%       % ��ֵΪ1   accuracy =82%
    %-t     % ��ֵΪ0   accuracy =73.5099%      % ��ֵΪ1   accuracy =49.8344%
            % ��ֵΪ2   accuracy =80.106%       % ��ֵΪ3   accuracy =81.4437%
%%  ***********************************************************************************   
    %  %�����д�ı����ı�
% data_s=S5_7();
    %-s     % ��ֵΪ0   accuracy =79.1457%     % ��ֵΪ1   accuracy =81.8325%
    %-t     % ��ֵΪ0   accuracy =75.8278%     % ��ֵΪ1   accuracy =74.6689%
            % ��ֵΪ2   accuracy =80%          % ��ֵΪ3   accuracy =78.1457%
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
