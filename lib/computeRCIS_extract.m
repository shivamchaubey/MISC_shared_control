function [H_struct, f_struct, volume] = computeRCIS_extract( ...
    A,B,Gx,Fx,Gu,Fu,E,Gw,Fw,implicit,L,T)

    if nargin < 12, T = []; end

    if ~isempty(T)
        [cis1, ~] = computeRCIS(A,B,Gx,Fx,Gu,Fu,E,Gw,Fw,implicit,L,T);
    else
        [cis1, ~] = computeRCIS(A,B,Gx,Fx,Gu,Fu,E,Gw,Fw,implicit,L);
    end

    % Disturbance set and Pontryagin difference
    W = Polyhedron('H', [Gw Fw]);
    cis1 = cis1 - E * W; % To handle disturbances

    % Extract H-reps and volumes
    H_struct = cell(1, numel(cis1));
    f_struct = cell(1, numel(cis1));
    volume   = zeros(1, numel(cis1));
    for i = 1:numel(cis1)
        cis1(i).minHRep();
        H = cis1(i).H;          % [F | f]
        H_struct{i} = H(:,1:end-1);
        f_struct{i} = H(:,end);
        volume(i)   = cis1(i).volume;
    end
end
