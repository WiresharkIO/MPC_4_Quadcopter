ar = [1, 2, 3]; % row vector

ac = [1; 2; 3]; % column vector

ci = eye(3); % Identity matrix of 3x3

appendhorizontal = [ar; ci];

appendvertical = [ac, ci];

artranspose = ar'; % transpose

temp=ar(1); % gets the element based on index

getsize = size(appendhorizontal);

createRowVector = 1:9; % creates row vector of given size in colon

onesmatrix = ones(2, 3);

zerosmatrix = zeros(2, 3);

randmatrix = rand(2, 3);

MLMarks = [1, 2, 3;
    2, 3, 4;
    1, 4, 2;
    3, 2, 1;
    5, 2, 1;
    1, 2, 2];

AImarks = [1, 2, 3, 4, 5, 6];

allMarks = [MLMarks, AImarks'];

fourthcolumnallMarks = allMarks(1:6, 4);

thirdrowallMarks = allMarks(3, :);

meanofall=mean(allMarks);

sumofallcolumns = sum(allMarks, 2); % 2 means sum across column, 1 will be
                                    % sum across row

diagforscaling = diag([0.1, 0.1, 0.1, 0.2]); 

scaledallMarks = allMarks*diagforscaling;
