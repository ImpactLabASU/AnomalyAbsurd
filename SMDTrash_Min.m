%% Trash Algorithm on SMD data preprocessing

clc 
clear all
trainDir = dir('../../UnknownUnknown/OmniAnomaly/ServerMachineDataset/train/');
testDir = dir('../../UnknownUnknown/OmniAnomaly/ServerMachineDataset/test/');
smoothWindow = 500;
TPMaster = 0;
TNMaster = 0;
FPMaster = 0;
FNMaster = 0;
eventNum = 0;

trainDir = dir('../../UnknownUnknown/OmniAnomaly/ServerMachineDataset/train/');
sizeEvent = [];
for j = 3:size(trainDir,1)

    dataM1 = dlmread(strcat([trainDir(j).folder '\' trainDir(j).name]));
    
    [index{j}, col] = readSMD(strcat(['../../UnknownUnknown/OmniAnomaly/ServerMachineDataset/interpretation_label/' trainDir(j).name ]));
    sizeEvent = [sizeEvent; abs(index{j}(:,1) - index{j}(:,2))];

end




for jM = 3:size(trainDir,1)
    valid = 40;
    lambda = 0.1;
    prevP = 0;

    %% Training
    
    %%
    


    dataM1 = dlmread(strcat([testDir(jM).folder '\' testDir(jM).name]));
    testLabels = dlmread(strcat(['../../UnknownUnknown/OmniAnomaly/ServerMachineDataset/test_label/' testDir(jM).name]));
    eventNum = eventNum + findContiguous(testLabels);
    predLabels = zeros(size(testLabels,1),1);
    % Validation strategy
%     for jK = 1:1000:size(dataM1,1)
%         validData = dataM1(jK:jK+floor(size(dataM1,1)*valid/100)-1,:);
%         validTest = testLabels(jK:jK+floor(size(dataM1,1)*valid/100)-1,1);
%         if(sum(validTest) > 0)
%             break;
%         end
%     end
    validData = dataM1(:,:);
    validTest = testLabels;
    ERR = abs(diff(dataM1(:,1)));
    [G,H] = find(ERR == 0);
    ERR(G) = [];
    e2M = 2;
    epsilon1 = min(ERR);
    epsilon2 = e2M*epsilon1;
    
    prevEp2 = epsilon2;
    prevEp1 = epsilon1;
%     kp = 0;
%     windowL = 500;
%     for jp = windowL:windowL:size(validData,1)
%         kp = kp + 1;
%         meanJ(kp) = mean(validData(jp-windowL+1:jp,9));
%     end
%     
%     AH = abs(diff(meanJ));
% 
%     [sortedAH] = sort(AH,'ascend');
%     epsilon1 = mean(sortedAH(1:2));
%     epsilon2 = epsilon1;
    Win = 2;
    predLabels = AnomalyTrash(validData,epsilon1,epsilon2,validTest, Win,1,1);
        % 
        [Accuracy, Precision, Recall, F1, TP, TN, FP, FN] = EvalAccV2(validTest,predLabels);
    prevP = F1;  
    a = (rand() > 0);
    if(a == 0)
        a = -1;
    end
    WinP = 2;
    for pp = 1:25
    for km = 1:100
        
            
        
            errP = 1 - F1;
            epsilon1 = epsilon1*rand();
            epsilon2 = epsilon2 + a*rand()*lambda*errP;
            Win = randi([1 2]);
            %epsilon2 = e2M*epsilon1;
            predLabels = AnomalyTrash(validData,epsilon1,epsilon2,validTest,Win,1,pp);
            [Accuracy, Precision, Recall, F1, TP, TN, FP, FN] = EvalAccV2(validTest,predLabels);
            if(F1 >= prevP)
                prevEp2 = epsilon2;
                prevEp1 = epsilon1;
                prevP = F1;
                WinP = Win;
                newPP(jM-2) = pp;
            else
                epsilon2 = prevEp2;
                epsilon1 = prevEp1;
                Win = WinP;
                a = -a;
            end

            


    end
    end

    predLabels = AnomalyTrash(dataM1,epsilon1,prevEp2,testLabels,WinP,1,newPP(jM-2));
    [Accuracy, Precision, Recall, F1, TP, TN, FP, FN] = EvalAccV2(testLabels,predLabels);

   TPMaster = TPMaster + TP;
   FPMaster = FPMaster + FP;
   FNMaster = FNMaster + FN;
   TNMaster = TNMaster + TN;
    
    
end

PrecisionM = TPMaster/(TPMaster+FPMaster)

RecallM = TPMaster/(TPMaster + FNMaster)

F1 = 2/(1/PrecisionM + 1/RecallM)



function [Accuracy, Precision, Recall, F1, TP, TN, FP, FN] = EvalAccV2(testLabels,predLabels)

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;

    for j = 1:max(size(testLabels))
        if(testLabels(j) == 1 && predLabels(j) == 1)

           TP = TP + 1;
        elseif(testLabels(j) == 0 && predLabels(j) == 1)
            FP = FP + 1;
        elseif(testLabels(j) == 1 && predLabels(j) == 0)
            FN = FN + 1;
        else
            TN = TN + 1;
        end
    end

    Precision = TP/(TP+FP);
    

    Recall = TP/(TP + FN);
    F1 = 2/(1/Precision + 1/Recall);
    Accuracy = (TP+TN)/(TP+TN+FP+FN);
    if(isnan(Precision))
        Precision = 0;
    end
    if(isnan(Recall))
        Recall = 0;
    end
    if(isnan(F1))
        F1 = 0;
    end
    if(isnan(Precision))
        Accuracy = 0;
    end
end

function predLabels = AnomalyTrash(dataM1, epsilon1,epsilon2,testLabels, windowL2,PA,pp)
    N = pp;
    anomalyP = [];
    
    searchAnom = 0;
    windowL = windowL2;
    numC = 0;
    predLabels = zeros(size(dataM1,1),1);
    for i = 2:size(dataM1,1)
        if(searchAnom == 0)
            if(abs(dataM1(i,N)-dataM1(i-1,N))<epsilon1)
                numC = numC + 1;
               
            else
                numC = 0;
            end
            if(numC > windowL)
                searchAnom = 1;
            end
        else
            if(abs(dataM1(i,N)-dataM1(i-1,N))>= epsilon2)
                anomalyP = [anomalyP i];
                searchAnom = 0;
                numC = 0;
            end
            
        end
    end
    
    predLabels(anomalyP) = 1;
    if(PA==1)
    anomaly_state = 0;
    for j = 1:size(testLabels,1)
        if(testLabels(j) == 1 && predLabels(j) == 1 && anomaly_state == 0)

            anomaly_state = 1;
            for k = j:-1:1
                if(testLabels(k) == 0)
                    break;
                else
                    if(predLabels(k) == 0)
                        predLabels(k) = 1;
                    end
                    
                end


            end

            for k = j:size(testLabels,1)
                if(testLabels(k) == 0)
                    break;
                else
                    if(predLabels(k) == 0)
                        predLabels(k) = 1;
                    end
                    
                end


            end


        elseif(testLabels(j) == 0)
            anomaly_state = 0;
        end

        if(anomaly_state == 1)
            predLabels(j) =1;
        end
        
    end
    end
end




function numSegments = findContiguous(A)

% Example array
array = A';

% Find the difference between consecutive elements
diffArray = diff([0, array, 0]);

% Find the starting points of contiguous segments of 1s
startIndices = find(diffArray == 1);

% Find the ending points of contiguous segments of 1s
endIndices = find(diffArray == -1) - 1;

% Number of contiguous segments of 1s
numSegments = length(startIndices);

% Display the result
disp(['Number of contiguous segments of 1s: ', num2str(numSegments)]);

end




function [index, col] = readSMD(filename)

% Open the file for reading
fid = fopen(filename, 'r');
if fid == -1
    error('File could not be opened.');
end

% Initialize variables to store results
rangeMatrix = [];
numberCells = {};

% Read the file line by line
while ~feof(fid)
    % Get the line of data
    line = fgetl(fid);
    
    % Split the line into the range part and the numbers part
    parts = strsplit(line, ':');
    
    % Get the range part and the numbers part
    rangePart = parts{1};
    numbersPart = parts{2};
    
    % Split the range part into the start and end numbers
    range = str2num(rangePart); %#ok<ST2NM>
    startEnd = str2double(strsplit(rangePart, '-'));
    
    % Split the numbers part into a list of numbers
    numbers = str2double(strsplit(numbersPart, ','));
    
    % Store the start and end numbers in the matrix
    rangeMatrix = [rangeMatrix; startEnd];
    
    % Store the numbers in the cell array as a column vector
    numberCells{end+1} = numbers(:);
end

% Close the file
fclose(fid);

index = rangeMatrix;

col = numberCells;

end