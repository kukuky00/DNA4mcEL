%% Species:   C.elegans ------Acc=78.04 %
% Benchmark dataset S1 contains 1554 4mC site containing sequences and 1554
% non-4mC site containing sequences of C. elegans.
close all;
clear;
clc;

%��һ��д�ı����ı�
data_s=S1_1();

%�ڶ���д�ı����ı�
% data_s=S1_2();

%������д�ı����ı�
% data_s=S1_3();

%���Ĵ�д�ı����ı�
% data_s=S1_4();

%�����д�ı����ı�
% data_s=S1_5();

%������д�ı����ı�
% data_s=S1_6();

%���ߴ�д�ı����ı�
% data_s=S1_7();

%��8��д�ı����ı�
% data_s=S1_8();

%��9��д�ı����ı�
% data_s=S1_9();
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
    model=svmtrain(training_label,training_data,'-s 1' );
%     model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
     
%%��һ��д�ı����ı�
% data_s=S1_1();
    %-s
    % ��ֵΪ0   accuracy_L =78.2819%      % ��ֵΪ1   accuracy =80.4054% 
    % ��ֵΪ2    accuracy =22.2973%       % ��ֵΪ3  and  % ��ֵΪ4  is wrong 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L = 81.3707%    % ��ֵΪ1   accuracy =72.3938% 
    % ��ֵΪ2    accuracy =77.8958%      % ��ֵΪ3    accuracy =76.5444%
    %-d �˺�������      accuracy = 77.4485
%%  ***********************************************************************************   
%%�ڶ���д�ı����ı�
% data_s=S1_2();
    %-s
    % ��ֵΪ0   accuracy_L =76.6409%       % ��ֵΪ1   accuracy = 81.0811% 
    % ��ֵΪ2    accuracy =23.8417%      % ��ֵΪ3  and  % ��ֵΪ4  is wrong 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =  78.7645%    % ��ֵΪ1   accuracy =46.7181% 
    % ��ֵΪ2   accuracy =76.9305%        ��ֵΪ3    accuracy =76.6409%
    
%%  ***********************************************************************************   
%%������д�ı����ı�
% data_s=S1_3();
    %-s
    % ��ֵΪ0   accuracy_L =78.0888%       % ��ֵΪ1   accuracy = 82.1776% 
    % ��ֵΪ2    accuracy = 23.3591%      % ��ֵΪ3  and  % ��ֵΪ4  is wrong 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L = 79.0541%      % ��ֵΪ1   accuracy = 47.7799%
    % ��ֵΪ2    accuracy = 78.3784%      % ��ֵΪ3    accuracy =75.5444%

%%  ***********************************************************************************   
%%���Ĵ�д�ı����ı�
% data_s=S1_4();
    %-s
    % ��ֵΪ0   accuracy_L = 78.861%    % ��ֵΪ1   accuracy =82.3359%
    % ��ֵΪ2    28.2819%     % ��ֵΪ3  and  % ��ֵΪ4  is wrong 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =79.2471%   % ��ֵΪ1   accuracy = 49.5174% 
    % ��ֵΪ2    accuracy =  79.7297%      % ��ֵΪ3    accuracy =78.0888% 
%%�����д�ı����ı�
% data_s=S1_5();
    %-s
    % ��ֵΪ0   accuracy_L = 79.0541%    % ��ֵΪ1   accuracy =82.3359% 
    % ��ֵΪ2    28.2819%     % ��ֵΪ3  and  % ��ֵΪ4  is wrong 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =77.5097%   % ��ֵΪ1   accuracy =  49.0347%
    % ��ֵΪ2    accuracy =  79.8263%      % ��ֵΪ3    accuracy =78.0888%     
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
