%% Species:   E.coli  ------Acc=79.82%
% Benchmark dataset S4 contains 388 4mC site containing sequences and 
% 388 non-4mC site containing sequences of E. coli.
close all;
clear;
clc;

%��һ��д�ı����ı�
% data_s=S4_1();

%�ڶ���д�ı����ı�
% data_s=S4_2();

%������д�ı����ı�
% data_s=S4_4();

%���Ĵ�д�ı����ı�
% data_s=S4_6();

%�����д�ı����ı�
data_s=S4_7();
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
% data_s=S3_1();
    %-s     % ��ֵΪ0   accuracy =  78.3784%       % ��ֵΪ1   accuracy =80.4574%
    %-t     % ��ֵΪ0   accuracy =74.4186%      % ��ֵΪ1   accuracy =48.2625%
            % ��ֵΪ2   accuracy =76       % ��ֵΪ3   accuracy =77
%%  ***********************************************************************************   
%%�ڶ���д�ı����ı�
% data_s=S3_2();
    %-s     % ��ֵΪ0   accuracy =79%      % ��ֵΪ1   accuracy =82.6255%
    %-t     % ��ֵΪ0   accuracy =77%      % ��ֵΪ1   accuracy = 46.7181% 
            % ��ֵΪ2   accuracy =78%     % ��ֵΪ3   accuracy =76% 
%%  ***********************************************************************************   
%%������д�ı����ı�
% data_s=S3_4();
    %-s     % ��ֵΪ0   accuracy =75%      % ��ֵΪ1   accuracy =81.4672%
    %-t     % ��ֵΪ0   accuracy =77%      % ��ֵΪ1   accuracy =46.332%
            % ��ֵΪ2   accuracy =79%      % ��ֵΪ3   accuracy =70%
%%  ***********************************************************************************   
%%���Ĵ�д�ı����ı�
% data_s=S3_6();
    %-s     % ��ֵΪ0   accuracy =80%        % ��ֵΪ1   accuracy =82%
    %-t     % ��ֵΪ0   accuracy =76.834%    % ��ֵΪ1   accuracy = 46.8992% 
            % ��ֵΪ2   accuracy =78%        % ��ֵΪ3   accuracy =78%
%%  ***********************************************************************************   
    %  %�����д�ı����ı�
% data_s=S3_7();
    %-s     % ��ֵΪ0   accuracy =79%         % ��ֵΪ1   accuracy = 80.2394% 
    %-t     % ��ֵΪ0   accuracy =76.4479%    % ��ֵΪ1   accuracy =47.8764%
            % ��ֵΪ2   accuracy =79%            % ��ֵΪ3   accuracy = 77%
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
