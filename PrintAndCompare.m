% Copyright (c) 2025 Martti Kesaniemi

function PrintAndCompare(str, v1, v2)
% Helper function to compare the output of Python and MATLAB implementations.

fprintf('\n\n%s:\n', str);
fprintf('%.2f ', v1);
fprintf('\n');
fprintf('%.2f ',v2);
fprintf('\n');
assert(all(abs(v2(:)-v1(:))<0.001))
end