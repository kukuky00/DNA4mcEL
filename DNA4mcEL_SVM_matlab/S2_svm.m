%% Species:  D.melanogaster ------Acc=81.16%
% Benchmark dataset S2 contains 1769 4mC site containing sequences and
% 1769 non-4mC site containing sequences of D. melanogaster.
close all;
clear;
clc;

%��һ��д�ı����ı�
% data_s=S2_1();

%�ڶ���д�ı����ı�
% data_s=S2_2();

%������д�ı����ı�
% data_s=S2_4();

%���Ĵ�д�ı����ı�
% data_s=S2_6();

%�����д�ı����ı�
data_s=S2_7();

%�����д�ı����ı�
% data_s=S2_8();
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
%   model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
     
%  %��һ��д�ı����ı�
% data_s=S2_1();
    %-s     % ��ֵΪ0   accuracy_L =81.1705%       % ��ֵΪ1   accuracy 82.6972%
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =82.4576%    % ��ֵΪ1   accuracy =78.2019% 
    % ��ֵΪ2    accuracy =81.9492%      % ��ֵΪ3    accuracy =81.2553%
    %-d �˺�������      accuracy = 77.4485
%%  ***********************************************************************************   
    %  %�ڶ���д�ı����ı�
% data_s=S2_2();
    %-s    % ��ֵΪ0   accuracy_L =81.6794%        % ��ֵΪ1   accuracy =82.6271% 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L = 80.5768%    % ��ֵΪ1   accuracy =73.3673%
    % ��ֵΪ2    accuracy = 81.1705%      % ��ֵΪ3    accuracy =79.7286% 
    
%%  ***********************************************************************************   
    %  %������д�ı����ı�
% data_s=S2_4();
    %-s    % ��ֵΪ0   accuracy_L =82.5454%       % ��ֵΪ1  82.1883% 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L = 79.1349%     % ��ֵΪ1   accuracy = 49.4487% 
    % ��ֵΪ2    accuracy = 82.2034%       % ��ֵΪ3    accuracy =81.2712%

%%  ***********************************************************************************   
    %  %���Ĵ�д�ı����ı�
% data_s=S2_6();
    %-s     % ��ֵΪ0   accuracy_L = 82.782%    % ��ֵΪ1   accuracy = 82.8668%
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =78.0322%      % ��ֵΪ1   accuracy = 49.4487%
    % ��ֵΪ2    accuracy =81.9338%       % ��ֵΪ3    accuracy =80.7464% 
%%  ***********************************************************************************   
    %  %���Ĵ�д�ı����ı�
% data_s=S2_6();
    %-s     % ��ֵΪ0   accuracy_L =81.3401%      % ��ֵΪ1   accuracy = 82.2034%
    %-t �˺�������
    % ��ֵΪ0   accuracy_L = 79.05%     % ��ֵΪ1   accuracy = 75.5085% 
    % ��ֵΪ2    accuracy =81.9492%        % ��ֵΪ3    accuracy = 80.8312% 

% data_s=S2_7();
    %-s     % ��ֵΪ0   accuracy_L =     % ��ֵΪ1   accuracy =82.2203% 
    %-t �˺�������
    % ��ֵΪ0   accuracy_L =   % ��ֵΪ1   accuracy =  
    % ��ֵΪ2    accuracy =      % ��ֵΪ3    accuracy =  

% data_s=S2_7();
    %-s     % ��ֵΪ0   accuracy_L =     % ��ֵΪ1   accuracy =83.1203% 
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
