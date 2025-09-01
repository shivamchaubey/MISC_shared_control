% clc;
close all;
clear all;
data = load('map8_safe_set.mat');


for i=1:size(data.cis_data,2)-1
    for j=1:2
        figure;
        hold on;
        Al = data.cis_data{i}{j}.Al;
        bl = data.cis_data{i}{j}.bl;
        Au = data.cis_data{i}{j}.Au;
        bu = data.cis_data{i}{j}.bu;
        keyname = fieldnames(bu);    
        for k=1:size(keyname,2)
            Al_value = getfield(Al, keyname{k});
            bl_value = getfield(bl, keyname{k});  
            Au_value = getfield(Au, keyname{k});
            bu_value = getfield(bu, keyname{k});
            if j == 1
                dim = [1 3]
            end
            if j == 2
                dim = [2 4]
            end
            if size(Al_value,1) >0
                poly_project_l = project(Al_value, bl_value, dim)
                plot(poly_project_l, label = 'lower')
            end

            if size(Au_value,1) > 0 
                poly_project_u = project(Au_value, bu_value, dim)
        
                plot(poly_project_u, label = 'upper')
        
            end
        end
    legend()
    hold off;
    end
    disp(i)
end



function poly_project=project(A, b, idx)
    size(A)
    poly = Polyhedron('A', A, 'b', b)
    poly_project = projection(poly, idx);
end





