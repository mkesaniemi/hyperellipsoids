% Copyright (c) 2025 Martti Kesaniemi

function [testsPassed, totalTests] = runTests()
% Some unit tests for hyperellipsoid fitting

    function [pass] = testInputSize()
        pass = false;

        e = []; try hyperellipsoidfit([]); catch e; end
        if isempty(e), return; end

        e = []; try hyperellipsoidfit(1); catch e; end
        if isempty(e), return; end

        e = []; try hyperellipsoidfit(magic(2)); catch e; end
        if ~isempty(e), return; end

        pass = true; % If all tests pass, set pass to true
    end

    function [pass] = testRegularization()
        pass = false;

        e = []; try S = hyperellipsoidfit([magic(2) magic(2)+0.1], 0); catch e; end
        if ~isempty(e), return; end
        if S.success, return; end
        
        e = []; try S = hyperellipsoidfit([magic(2) magic(2)+0.1], 0.3); catch e; end
        if ~isempty(e), return; end       
        if ~S.success, return; end

        pass = true; % If all tests pass, set pass to true
    end

    function [pass] = testNormalization()
        points = table2array(combinations([-1 1], [-1 1], [-1 1])) + 10;
        pass = false;

        e = [];
        try S = hyperellipsoidfit(points, 0, 'SOD', 'normalize', true);
        catch e
        end
        if ~isempty(e), return; end
        if S.success, return; end
        if sum(abs(S.A) > eps) > 2
            return;
        end

        e = [];
        try S = hyperellipsoidfit(points, 0, 'SOD', 'normalize', false);
        catch e
        end
        if ~isempty(e), return; end       
        if ~S.success, return; end
        if sum(abs(S.A) > eps) < 6
            return;
        end

        pass = true; % If all tests pass, set pass to true
    end


testsPassed = 0;
testsFailed = 0;

if ~testInputSize()
    testsFailed = testsFailed + 1;
else
    testsPassed = testsPassed + 1;
end

if ~testRegularization()
    testsFailed = testsFailed + 1;
else
    testsPassed = testsPassed + 1;
end

if ~testNormalization()
    testsFailed = testsFailed + 1;
else
    testsPassed = testsPassed + 1;
end

totalTests = testsPassed + testsFailed; 

fprintf('%d out of %d tests passed.', testsPassed, totalTests);

end
