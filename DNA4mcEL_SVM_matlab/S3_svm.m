%% Species:  A.thaliana  ------Acc=76.05 
% Benchmark dataset S3 contains 1978 4mC site containing sequences and
% 1978 non-4mC site containing sequences of A.thaliana.
close all;
clear;
clc;

%��һ��д�ı����ı�
% data_s=S3_1();

%�ڶ���д�ı����ı�
% data_s=S3_2();

%������д�ı����ı�
% data_s=S3_4();

%���Ĵ�д�ı����ı�
% data_s=S3_6();

%�����д�ı����ı�
% data_s=S3_7();

%��6��д�ı����ı�
data_s=S3_8();
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
    model=svmtrain(training_label,training_data,'-t 1' );
%   model=svmtrain(training_label,training_data,'-c 1 -g 0.07' ); %Accuracy = 52.4131% 
%%      model=svmtrain(training_label,training_data,'-t 0 -v 10');
     
%%��һ��д�ı����ı�
% data_s=S3_1();
    %-s     % ��ֵΪ0   accuracy =76.4973%       % ��ֵΪ1   accuracy =76.8764% 
    %-t     % ��ֵΪ0   accuracy =76.8764%       % ��ֵΪ1   accuracy =74.2229% 
            % ��ֵΪ2   accuracy =76.3457%       % ��ֵΪ3   accuracy =76.3457% 
%%  ***********************************************************************************   
%%�ڶ���д�ı����ı�
% data_s=S3_2();
    %-s     % ��ֵΪ0   accuracy =77.1797%      % ��ֵΪ1   accuracy =78.2246%
    %-t     % ��ֵΪ0   accuracy =75.4359%      % ��ֵΪ1   accuracy =69.674%
            % ��ֵΪ2   accuracy =76.7071%      % ��ֵΪ3   accuracy =76.4215% 
%%  ***********************************************************************************   
%%������д�ı����ı�
% data_s=S3_4();
    %-s     % ��ֵΪ0   accuracy=76.5732%    % ��ֵΪ1   accuracy=77.6176%
    %-t     % ��ֵΪ0   accuracy=74.6778%   % ��ֵΪ1   accuracy =48.786%
            % ��ֵΪ2   accuracy =77.1039%    % ��ֵΪ3    accuracy =76.649%
%%  ***********************************************************************************   
%%���Ĵ�д�ı����ı�
% data_s=S3_6();
    %-s     % ��ֵΪ0   accuracy =77.4829%      % ��ֵΪ1   accuracy=77.0281%
    %-t     % ��ֵΪ0   accuracy = 73.8438%    % ��ֵΪ1   accuracy =49.9621%
            % ��ֵΪ2   accuracy =77.0281%   % ��ֵΪ3    accuracy =76.2699% 
%%  ***********************************************************************************   
    %  %�����д�ı����ı�
% data_s=S3_7();
    %-s     % ��ֵΪ0   accuracy = 77.1624%       % ��ֵΪ1   accuracy=77.7248%
    %-t     % ��ֵΪ0   accuracy =74.0516%   % ��ֵΪ1   accuracy =68.84%
            % ��ֵΪ2   accuracy =76.649%    % ��ֵΪ3    accuracy =75.8908%
    [predict_label,accuracy,dec_values]=svmpredict(tseting_lable,tseting_data,model);

   %%
  
end
