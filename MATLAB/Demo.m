% Copyright (c) 2025 Martti Kesaniemi

function Demo
% Demontrates the results gained with hyperellipsoidfit.m

methods = {'SOD', 'HES', ...
    'BOOK', 'FC', 'TAUB', '2-NORM'};

% Create data for visualization
[X, Y, Z] = sphere(30);
data = cat(3, X, Y, Z);
datasize = size(data);
data = reshape(data, [prod(datasize(1:2)), 3]);

% Create data for fitting
points = table2array(combinations([-1 1], [-1 1], [-1 1]));

% Initialize
figure(1);
clf;
offset = zeros(3, 3);
M = zeros(3,3,3);
s = false(1,3);
reg = zeros(1,3);
% Initialize handles
h_m = zeros(3,1); h_p = h_m; h_a = h_m; h_t = h_m; h_s = h_m;
firstdraw = true;
max_z = 30;
maxInd = numel(methods);

w = warning ('off','all');

for z = 0:0.02:(max_z-0.01)    
    % Choose method
    ind = z/max_z*maxInd;
    method  = methods{ceil(ind+eps)};
    axial   = ind - floor(ind) > .5;
    origin  = 2*ind - floor(2*ind) > .5;
    paramstr = '';
    if axial, paramstr = strcat(paramstr, ', ''forceAxial'', 1'); end
    if origin, paramstr = strcat(paramstr, ', ''forceOrigin'', 1'); end
    titlestr = {{' ',' ','R = %.2f'}, ...
        {sprintf(['\\bf hyperellipsoidfit(data, ' '''', method, '''' ', R' ...
        '%s);'], paramstr), ...
        '\rm ', 'R = %.2f'}, {' ',' ','R = %.2f'}};
    
    % Create additional moving point
    apoint = [3*cos(0.8*z*pi) .5*cos(0.6*z*pi) 4*sin(z*pi)];
    allpoints = cat(1, points, apoint);
    % Perform fittings
    [M(:,:,1), offset(:,1), s(1), ~, reg(1)] = ...
        hyperellipsoidfit(allpoints, 0.01, method, ...
        'forceOrigin', origin, ...
        'forceAxial', axial);
    [M(:,:,2), offset(:,2), s(2), ~, reg(2)] = ...
        hyperellipsoidfit(allpoints, 0.1, method, ...
        'forceOrigin', origin, ...
        'forceAxial', axial);
    [M(:,:,3), offset(:,3), s(3), ~, reg(3)] = ...
        hyperellipsoidfit(allpoints, 0.7, method, ...
        'forceOrigin', origin, ...
        'forceAxial', axial);
    for ii=1:3
        outData = bsxfun(@plus, (M(:,:,ii)*data')', offset(:,ii)');
        X1 = reshape(outData(:,1), datasize(1:2));
        Y1 = reshape(outData(:,2), datasize(1:2));
        Z1 = reshape(outData(:,3), datasize(1:2));
        
        % Visualize
        if firstdraw
            h_s(ii) = subplot(1,3,ii);
            p = get(h_s(ii), 'Position');
            p(3) = 0.27;
            set(h_s(ii), 'Position', p);
            h_m(ii) = mesh(X1, Y1, Z1, 'FaceAlpha', 0.00, ...
                'EdgeAlpha', 0.5, ...
                'EdgeColor', [0.0 0.5 0.0]);            
            hold on
            h_p(ii) = plot3(points(:,1),points(:,2),points(:,3), ...
                'ob','linewidth',1.5);
            h_a(ii) = plot3(apoint(:,1),apoint(:,2),apoint(:,3), ...
                'or','linewidth',2);
            hold off
            axis equal
            axis([-1 1 -1 1 -1.35 1.35]*3)
            axis off
            h_t(ii) = title(cat(2, titlestr{ii}(1:end-1), ...
                {sprintf(char(titlestr{ii}{end}),reg(ii))}));
        else
            set(h_m(ii), {'XData', 'YData', 'ZData'}, ...
                {X1, Y1, Z1});
            set(h_p(ii), {'XData', 'YData', 'ZData'}, ...
                {points(:,1), points(:,2), points(:,3)});
            set(h_a(ii), {'XData', 'YData', 'ZData'}, ...
                {apoint(:,1), apoint(:,2), apoint(:,3)});
            set(h_t(ii), 'String', ...
                cat(2, titlestr{ii}(1:end-1), ...
                {sprintf(char(titlestr{ii}{end}),reg(ii))}));
        end
    end
    drawnow;
    firstdraw = false;
end

warning (w);
