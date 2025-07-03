% Copyright (c) 2025 Martti Kesäniemi
% A script to compare the output of Python and MATLAB implementations.

fitpath = [pwd '\Python'];
if count(py.sys.path, fitpath) == 0
    insert(py.sys.path,int32(0), fitpath)
end
% Reload python code
clear classes
pythonfit = py.importlib.import_module('hyperellipsoidfit');
py.importlib.reload(pythonfit);

numSamples =30;
numDims = 9;
D = randn(numSamples, numDims) + [-1 10 zeros(1, numDims-2)];

addpath("MATLAB\");

for method_ = {'SOD', 'HES'}
    for forceOrigin_ = [false, true]
        for forceAxial_ = [false, true]
            disp('********************')
            [Me, oe, success, A, regParam, dist] = ...
                hyperellipsoidfit(D, [], method_{1}, ...
                'forceorigin', forceOrigin_, ...
                'forceaxial', forceAxial_);

            S = py.hyperellipsoidfit.hyperellipsoidfit(D, method = method_{1}, ...
                forceOrigin = forceOrigin_, forceAxial = forceAxial_);

            PrintAndCompare('success', double(success), double(S{3}))
            if success
                PrintAndCompare('Me', Me', double(S{1}))
                PrintAndCompare('oe', oe', double(S{2}))
                PrintAndCompare('A', A', double(S{4}))
                PrintAndCompare('regparam', regParam, double(S{5}))
                PrintAndCompare('dist', dist', double(S{6}))
            end
            disp('* * * * * * * * * *')
        end
    end
end